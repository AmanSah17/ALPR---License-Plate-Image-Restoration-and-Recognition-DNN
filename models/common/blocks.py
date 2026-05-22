"""
Residual and lightweight Swin-style transformer blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common.layers import ConvGNAct


class ResidualBlock(nn.Module):
    """Two-layer residual conv block with skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(channels, channels),
            ConvGNAct(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SwinTransformerBlock(nn.Module):
    """
    Lightweight windowed self-attention block (SwinIR-inspired).

    Uses small windows suitable for RLPR plate crops on GTX 1650.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def _partition(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B,C,H,W)`` to window tokens ``(B*nW, w*w, C)``."""
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, hp, wp = x.shape
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        tokens = x.view(-1, ws * ws, c)
        return tokens

    def _reverse(self, tokens: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        ws = self.window_size
        hp = h + (ws - h % ws) % ws
        wp = w + (ws - w % ws) % ws
        nwh = hp // ws
        nww = wp // ws
        c = tokens.shape[-1]
        x = tokens.view(b, nwh, nww, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        shortcut = x
        tokens = self._partition(x)
        tokens = self.norm1(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        x = self._reverse(tokens, b, h, w)
        return x + shortcut
