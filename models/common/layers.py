"""
Reusable convolutional layers for restoration networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    """Conv2d + GroupNorm + activation (VRAM-friendly vs BatchNorm)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups_norm: int = 8,
        act: nn.Module = None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        gn_groups = min(groups_norm, out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.act = act if act is not None else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Downsample(nn.Module):
    """Stride-2 downsample with conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = ConvGNAct(channels, channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """2x bilinear upsample + conv refine (preserves channel count for skips)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvGNAct(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))
