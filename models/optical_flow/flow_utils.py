"""
Optical flow tensor utilities: resizing for RAFT, magnitude, scaling, I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ResizeMeta:
    """Metadata to map RAFT-resolution flow back to original frame size."""

    orig_h: int
    orig_w: int
    raft_h: int
    raft_w: int
    scale_y: float
    scale_x: float


class FlowUtils:
    """Static helpers for flow magnitude and validation."""

    @staticmethod
    def flow_magnitude(flow: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel flow magnitude.

        Args:
            flow: ``(B, 2, H, W)`` or ``(2, H, W)``.

        Returns:
            Magnitude map ``(B, 1, H, W)`` or ``(1, H, W)``.
        """
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)
        return torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2 + 1e-8)

    @staticmethod
    def temporal_consistency(flows: torch.Tensor) -> float:
        """
        Mean L1 difference between consecutive flow fields.

        Args:
            flows: ``(T, 2, H, W)`` tensor.

        Returns:
            Scalar consistency score (lower = more consistent).
        """
        if flows.shape[0] < 2:
            return 0.0
        diffs = [
            torch.mean(torch.abs(flows[i + 1] - flows[i])).item()
            for i in range(flows.shape[0] - 1)
        ]
        return float(np.mean(diffs))


def resize_for_raft(
    image: torch.Tensor,
    min_spatial_size: int = 128,
    pad_to_multiple: int = 8,
) -> Tuple[torch.Tensor, ResizeMeta]:
    """
    Upscale and pad a single image for RAFT (small RLPR crops need this).

    Args:
        image: ``(3, H, W)`` float tensor in [0, 1].
        min_spatial_size: Minimum H and W after resize.
        pad_to_multiple: Pad dimensions to this multiple.

    Returns:
        Resized tensor ``(3, H', W')`` and ``ResizeMeta``.
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image.shape)}")

    _, h, w = image.shape
    scale = max(min_spatial_size / h, min_spatial_size / w, 1.0)
    new_h = int(np.ceil(h * scale))
    new_w = int(np.ceil(w * scale))

    if pad_to_multiple > 1:
        new_h = int(np.ceil(new_h / pad_to_multiple) * pad_to_multiple)
        new_w = int(np.ceil(new_w / pad_to_multiple) * pad_to_multiple)

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    meta = ResizeMeta(
        orig_h=h,
        orig_w=w,
        raft_h=new_h,
        raft_w=new_w,
        scale_y=new_h / h,
        scale_x=new_w / w,
    )
    return resized, meta


def scale_flow_to_original(flow: torch.Tensor, meta: ResizeMeta) -> torch.Tensor:
    """
    Scale RAFT flow from RAFT resolution back to original resolution.

    Args:
        flow: ``(2, H', W')`` at RAFT resolution.
        meta: Resize metadata from ``resize_for_raft``.

    Returns:
        Flow ``(2, H, W)`` at original resolution.
    """
    # Flow vectors scale with pixel coordinate scaling
    flow_scaled = flow.clone()
    flow_scaled[0] = flow_scaled[0] / meta.scale_x
    flow_scaled[1] = flow_scaled[1] / meta.scale_y

    flow_orig = F.interpolate(
        flow_scaled.unsqueeze(0),
        size=(meta.orig_h, meta.orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return flow_orig


def numpy_hwc_to_chw_float(image: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC RGB to float CHW [0,1]."""
    arr = image.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))).contiguous()


def flow_to_numpy(flow: torch.Tensor) -> np.ndarray:
    """Detach flow ``(2,H,W)`` to float32 numpy."""
    return flow.detach().cpu().numpy().astype(np.float32)
