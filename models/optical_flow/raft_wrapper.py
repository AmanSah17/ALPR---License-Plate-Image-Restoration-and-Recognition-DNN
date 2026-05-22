"""
RAFT-small wrapper using torchvision pretrained weights (FP16, low VRAM).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from models.optical_flow.flow_utils import ResizeMeta, resize_for_raft, scale_flow_to_original
from utils.gpu_utils import GPUManager

logger = logging.getLogger(__name__)

try:
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchvision optical_flow RAFT requires torchvision>=0.16"
    ) from exc


class RAFTSmallWrapper(nn.Module):
    """
    Pretrained RAFT-small with RLPR-friendly upscaling for small plate crops.

    Computes flow from ``image1`` to ``image2`` (both CHW float [0,1]).
    """

    def __init__(
        self,
        pretrained: bool = True,
        mixed_precision: bool = True,
        min_spatial_size: int = 128,
        pad_to_multiple: int = 8,
        gpu_manager: Optional[GPUManager] = None,
    ) -> None:
        """
        Args:
            pretrained: Load torchvision default weights.
            mixed_precision: Use autocast on CUDA.
            min_spatial_size: Upscale short side before RAFT.
            pad_to_multiple: Pad H/W for RAFT encoder.
            gpu_manager: Optional GPU helper for device/autocast.
        """
        super().__init__()
        self.mixed_precision = mixed_precision
        self.min_spatial_size = min_spatial_size
        self.pad_to_multiple = pad_to_multiple
        self.gpu = gpu_manager or GPUManager()

        weights = Raft_Small_Weights.DEFAULT if pretrained else None
        self.model = raft_small(weights=weights)
        self.transforms = (
            weights.transforms() if weights is not None else Raft_Small_Weights.DEFAULT.transforms()
        )
        self.model = self.gpu.move_module(self.model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(
            "RAFT-small loaded (pretrained=%s, device=%s, min_size=%d)",
            pretrained,
            self.gpu.device,
            min_spatial_size,
        )

    @torch.inference_mode()
    def predict_pair(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> Tuple[torch.Tensor, ResizeMeta]:
        """
        Estimate optical flow from image1 -> image2 at original resolution.

        Args:
            image1: ``(3, H, W)`` float [0, 1].
            image2: ``(3, H, W)`` float [0, 1].

        Returns:
            Flow ``(2, H, W)`` and resize metadata.
        """
        if image1.shape != image2.shape:
            raise ValueError(f"Image shape mismatch: {image1.shape} vs {image2.shape}")

        img1_r, meta = resize_for_raft(
            image1, self.min_spatial_size, self.pad_to_multiple
        )
        img2_r, _ = resize_for_raft(
            image2, self.min_spatial_size, self.pad_to_multiple
        )

        # torchvision transforms expect batch CHW
        t1, t2 = self.transforms(img1_r.unsqueeze(0), img2_r.unsqueeze(0))
        t1 = self.gpu.to_device(t1)
        t2 = self.gpu.to_device(t2)

        with self.gpu.autocast_context(enabled=self.mixed_precision):
            flow_list = self.model(t1, t2)

        # RAFT returns list of refinements; last is finest
        flow_raft = flow_list[-1][0]  # (2, H', W') on GPU
        flow_orig = scale_flow_to_original(flow_raft, meta).cpu()
        self.gpu.maybe_empty_cache()
        return flow_orig, meta

    @torch.inference_mode()
    def predict_batch_pairs(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Sequentially predict flows for a list of pairs (VRAM-safe).

        Args:
            pairs: List of (image1, image2) CHW tensors.

        Returns:
            List of flows ``(2, H, W)``.
        """
        flows: List[torch.Tensor] = []
        for img1, img2 in pairs:
            flow, _ = self.predict_pair(img1, img2)
            flows.append(flow)
        return flows

    @classmethod
    def from_config(cls, cfg: Any, gpu_manager: Optional[GPUManager] = None) -> "RAFTSmallWrapper":
        """Build from ``optical_flow`` config section."""
        section = cfg.get("optical_flow", cfg) if hasattr(cfg, "get") else cfg
        return cls(
            pretrained=bool(section.get("pretrained", True)),
            mixed_precision=bool(section.get("mixed_precision", True)),
            min_spatial_size=int(section.get("min_spatial_size", 128)),
            pad_to_multiple=int(section.get("pad_to_multiple", 8)),
            gpu_manager=gpu_manager,
        )
