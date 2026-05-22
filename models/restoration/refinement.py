"""
OCR-aware refinement hook (edge-emphasis) for character stroke recovery.

Full PARSeq integration arrives in Phase 6; this module pre-emphasizes edges
for sharper glyphs before OCR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.layers import ConvGNAct


class OCRRefinementHook(nn.Module):
    """
    Optional edge-guided feature refinement.

    When enabled, blends encoder features with Sobel edge features from input RGB.
    """

    def __init__(self, channels: int, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled
        if not enabled:
            self.register_module("fuse", None)
            return
        self.edge_conv = nn.Conv2d(1, channels, kernel_size=1)
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1)

    @staticmethod
    def _sobel_edges(image: torch.Tensor) -> torch.Tensor:
        """Grayscale Sobel magnitude from ``(B,3,H,W)`` -> ``(B,1,H,W)``."""
        gray = image.mean(dim=1, keepdim=True)
        kx = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            dtype=gray.dtype,
            device=gray.device,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            dtype=gray.dtype,
            device=gray.device,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.fuse is None:
            return features
        edges = self._sobel_edges(image)
        edge_feat = self.edge_conv(edges)
        return self.fuse(torch.cat([features, edge_feat], dim=1))
