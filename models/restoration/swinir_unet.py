"""
Lightweight SwinIR encoder + UNet decoder hybrid for license plate restoration.

Optimized for 4GB VRAM: small channel width, few blocks, gradient checkpointing optional.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.blocks import ResidualBlock, SwinTransformerBlock
from models.common.layers import ConvGNAct, Downsample, Upsample
from models.restoration.refinement import OCRRefinementHook

logger = logging.getLogger(__name__)


class SwinIRUNetHybrid(nn.Module):
    """
    Encoder-decoder with Swin-style blocks and skip connections.

    Forward uses global residual learning: ``output = input + network(input)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 4,
        window_size: int = 4,
        num_heads: int = 4,
        use_skip_connections: bool = True,
        ocr_refinement_enabled: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip_connections
        self.gradient_checkpointing = gradient_checkpointing

        self.stem = ConvGNAct(in_channels, base_channels)
        self.enc1 = nn.Sequential(
            *[
                SwinTransformerBlock(base_channels, window_size, num_heads)
                for _ in range(num_blocks // 2)
            ],
            ResidualBlock(base_channels),
        )
        self.down = Downsample(base_channels)
        self.enc2 = nn.Sequential(
            *[
                SwinTransformerBlock(base_channels, window_size, num_heads)
                for _ in range(max(1, num_blocks - num_blocks // 2))
            ],
            ResidualBlock(base_channels),
        )
        self.bottleneck = ResidualBlock(base_channels)
        self.up = Upsample(base_channels)
        self.dec1 = nn.Sequential(
            ConvGNAct(base_channels, base_channels),
            ResidualBlock(base_channels),
        )
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.ocr_hook = OCRRefinementHook(base_channels, enabled=ocr_refinement_enabled)

    def _run_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restore input tensor.

        Args:
            x: ``(B, 3, H, W)`` in [0, 1].

        Returns:
            Restored tensor same shape, clamped to [0, 1].
        """
        inp = x
        s0 = self._run_checkpoint(self.stem, x)
        s1 = self._run_checkpoint(self.enc1, s0)
        s2 = self._run_checkpoint(self.down, s1)
        s2 = self._run_checkpoint(self.enc2, s2)
        b = self._run_checkpoint(self.bottleneck, s2)
        d = self._run_checkpoint(self.up, b)
        if self.use_skip:
            # Match odd/even spatial sizes after down/up (variable RLPR H×W)
            if d.shape[-2:] != s1.shape[-2:]:
                d = F.interpolate(d, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            d = d + s1
        d = self._run_checkpoint(self.dec1, d)
        d = self.ocr_hook(d, inp)
        residual = self.head(d)
        out = inp + residual
        return torch.clamp(out, 0.0, 1.0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: Any) -> "SwinIRUNetHybrid":
        r = cfg.restoration if hasattr(cfg, "restoration") else cfg
        ocr = r.ocr_refinement if hasattr(r, "ocr_refinement") else {}
        hw = cfg.hardware if hasattr(cfg, "hardware") else {}
        return cls(
            base_channels=int(r.get("base_channels", 32)),
            num_blocks=int(r.get("num_blocks", 4)),
            window_size=int(r.get("window_size", 4)),
            num_heads=int(r.get("num_heads", 4)),
            use_skip_connections=bool(r.get("use_skip_connections", True)),
            ocr_refinement_enabled=bool(ocr.get("enabled", False)),
            gradient_checkpointing=bool(hw.get("gradient_checkpointing", False)),
        )
