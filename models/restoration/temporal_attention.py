"""
Lightweight temporal attention fusion for multi-frame plate sequences.

Combines per-frame appearance features with optional flow-confidence priors
to produce a single sharpened latent frame before SwinIR-UNet restoration (Phase 5).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.optical_flow.flow_utils import FlowUtils


class TemporalAttentionFusion(nn.Module):
    """
    Frame-level attention with spatial gating (low VRAM).

    Pipeline per forward pass:
        1. Encode each frame with a tiny conv trunk.
        2. Spatial gate highlights plate strokes.
        3. Global pool -> per-frame score -> softmax weights.
        4. Optional flow-confidence prior from Phase 3.
        5. Weighted sum + small reference residual.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 16,
        num_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = max(temperature, 1e-4)
        self.num_heads = max(1, num_heads)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.spatial_gate = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # Multi-head scoring: split hidden dim into heads, average scores
        self.frame_scorers = nn.ModuleList(
            [nn.Linear(hidden_channels, 1) for _ in range(self.num_heads)]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ref_blend = nn.Parameter(torch.tensor(0.25))

    def _encode_frames(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pooled descriptors ``(T, C)`` and spatial gates ``(T,1,H,W)``."""
        feats = self.encoder(frames)
        spatial = torch.sigmoid(self.spatial_gate(feats))
        denom = spatial.sum(dim=(2, 3)).clamp(min=1e-6)
        pooled = (feats * spatial).sum(dim=(2, 3)) / denom
        return pooled, spatial

    def _frame_weights(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Compute softmax weights ``(T,)`` from descriptors ``(T, C)``."""
        scores = []
        for head in self.frame_scorers:
            scores.append(head(descriptors).squeeze(-1))
        score = torch.stack(scores, dim=0).mean(dim=0)
        score = self.dropout(score)
        return F.softmax(score / self.temperature, dim=0)

    def forward(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        flows: Optional[torch.Tensor] = None,
        use_flow_confidence: bool = True,
        confidence_power: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse warped frames toward the reference coordinate system.

        Args:
            frames: ``(T, 3, H, W)``.
            reference: ``(3, H, W)``.
            flows: ``(T, 2, H, W)`` optional.

        Returns:
            fused, frame_weights ``(T,)``, spatial maps ``(T,1,H,W)``.
        """
        desc, spatial = self._encode_frames(frames)
        weights = self._frame_weights(desc)

        if use_flow_confidence and flows is not None and flows.shape[0] == frames.shape[0]:
            mags = FlowUtils.flow_magnitude(flows).view(frames.shape[0], -1).mean(dim=1)
            conf = torch.exp(-confidence_power * mags.to(weights.device))
            weights = weights * conf
            weights = weights / weights.sum().clamp(min=1e-8)

        fused = torch.sum(frames * weights.view(-1, 1, 1, 1), dim=0)
        fused = fused + torch.sigmoid(self.ref_blend) * reference
        fused = torch.clamp(fused, 0.0, 1.0)
        return fused, weights, spatial
