"""
Differentiable frame warping using estimated optical flow.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FrameWarper:
    """
    Warp source frames toward a reference using dense optical flow.

    Uses ``grid_sample`` with flow defined in pixel units (dx, dy).
    """

    def __init__(self, align_corners: bool = True) -> None:
        self.align_corners = align_corners

    @staticmethod
    def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel flow ``(2, H, W)`` to normalized ``grid_sample`` grid.

        Args:
            flow: Optical flow (x horizontal, y vertical).

        Returns:
            Grid ``(1, H, W, 2)`` in [-1, 1].
        """
        _, h, w = flow.shape
        # Base identity grid in pixel coords
        ys, xs = torch.meshgrid(
            torch.arange(h, device=flow.device, dtype=flow.dtype),
            torch.arange(w, device=flow.device, dtype=flow.dtype),
            indexing="ij",
        )
        x_map = xs + flow[0]
        y_map = ys + flow[1]
        # Normalize to [-1, 1]
        x_norm = 2.0 * x_map / max(w - 1, 1) - 1.0
        y_norm = 2.0 * y_map / max(h - 1, 1) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1)
        return grid.unsqueeze(0)

    def warp(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Warp ``image`` using ``flow`` (image1 -> image2 convention).

        Args:
            image: ``(3, H, W)`` source frame.
            flow: ``(2, H, W)`` flow at same resolution.
            mode: ``grid_sample`` interpolation mode.

        Returns:
            Warped image ``(3, H, W)``.
        """
        if image.shape[-2:] != flow.shape[-2:]:
            raise ValueError(f"Shape mismatch image {image.shape} flow {flow.shape}")
        grid = self._flow_to_grid(flow)
        warped = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=self.align_corners,
        )
        return warped.squeeze(0)

    def warp_error_map(
        self,
        reference: torch.Tensor,
        warped: torch.Tensor,
    ) -> torch.Tensor:
        """
        Absolute error map between reference and warped frame.

        Returns:
            Single-channel error ``(1, H, W)`` in [0, 1].
        """
        err = torch.mean(torch.abs(reference - warped), dim=0, keepdim=True)
        if err.max() > 0:
            err = err / err.max()
        return err

    def warp_batch(
        self,
        images: torch.Tensor,
        flows: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp a batch of images.

        Args:
            images: ``(T, 3, H, W)``.
            flows: ``(T, 2, H, W)``.

        Returns:
            Warped ``(T, 3, H, W)``.
        """
        outputs = [self.warp(images[i], flows[i]) for i in range(images.shape[0])]
        return torch.stack(outputs, dim=0)
