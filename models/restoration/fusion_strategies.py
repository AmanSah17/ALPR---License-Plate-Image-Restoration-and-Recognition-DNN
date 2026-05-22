"""
Temporal fusion strategies: mean, flow-confidence weighted, and attention-based.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from models.optical_flow.flow_utils import FlowUtils
from models.restoration.temporal_attention import TemporalAttentionFusion

logger = logging.getLogger(__name__)

FusionMethod = Literal["mean", "weighted", "attention"]


@dataclass
class FusionResult:
    """Output of a fusion strategy."""

    fused: torch.Tensor  # (3, H, W)
    weights: torch.Tensor  # (T,) frame weights (sum to 1)
    method: str
    extras: Dict[str, torch.Tensor]


class BaseFusionStrategy(ABC):
    """Abstract multi-frame fusion operator."""

    name: str = "base"

    @abstractmethod
    def fuse(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        flows: Optional[torch.Tensor] = None,
    ) -> FusionResult:
        """
        Fuse warped frames aligned to reference.

        Args:
            frames: ``(T, 3, H, W)`` warped frames (reference may be included).
            reference: ``(3, H, W)`` center / anchor frame.
            flows: Optional ``(T, 2, H, W)`` flows toward reference for confidence.

        Returns:
            ``FusionResult`` with fused image and per-frame weights.
        """


class MeanFusion(BaseFusionStrategy):
    """Uniform average over all warped frames."""

    name = "mean"

    def fuse(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        flows: Optional[torch.Tensor] = None,
    ) -> FusionResult:
        t = frames.shape[0]
        weights = torch.full((t,), 1.0 / t, device=frames.device, dtype=frames.dtype)
        fused = frames.mean(dim=0)
        return FusionResult(fused=fused, weights=weights, method=self.name, extras={})


class ConfidenceWeightedFusion(BaseFusionStrategy):
    """
    Weight frames by flow-confidence: lower motion magnitude -> higher weight.

    Uses ``exp(-power * mean_flow_magnitude)`` per frame.
    """

    name = "weighted"

    def __init__(self, power: float = 1.0, temperature: float = 1.0) -> None:
        self.power = power
        self.temperature = temperature

    def fuse(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        flows: Optional[torch.Tensor] = None,
    ) -> FusionResult:
        t = frames.shape[0]
        if flows is None or flows.shape[0] != t:
            logger.warning("No flows for weighted fusion; falling back to mean.")
            return MeanFusion().fuse(frames, reference, flows)

        mags = FlowUtils.flow_magnitude(flows).view(t, -1).mean(dim=1)
        scores = torch.exp(-self.power * mags)
        weights = F.softmax(scores / self.temperature, dim=0)
        fused = torch.sum(frames * weights.view(t, 1, 1, 1), dim=0)
        return FusionResult(
            fused=fused,
            weights=weights,
            method=self.name,
            extras={"flow_magnitude": mags},
        )


class AttentionFusionStrategy(BaseFusionStrategy):
    """Learned lightweight temporal attention (inference-only random init or trainable later)."""

    name = "attention"

    def __init__(
        self,
        hidden_channels: int = 16,
        num_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 1.0,
        use_flow_confidence: bool = True,
        confidence_power: float = 1.0,
    ) -> None:
        self.module = TemporalAttentionFusion(
            in_channels=3,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature,
        )
        self.use_flow_confidence = use_flow_confidence
        self.confidence_power = confidence_power

    def fuse(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        flows: Optional[torch.Tensor] = None,
    ) -> FusionResult:
        self.module.eval()
        with torch.no_grad():
            fused, attn_weights, spatial_map = self.module(
                frames, reference, flows, self.use_flow_confidence, self.confidence_power
            )
        return FusionResult(
            fused=fused,
            weights=attn_weights,
            method=self.name,
            extras={"spatial_attention": spatial_map},
        )


def build_fusion_strategy(
    method: FusionMethod,
    cfg: Optional[dict] = None,
) -> BaseFusionStrategy:
    """
    Factory for fusion strategies.

    Args:
        method: ``mean``, ``weighted``, or ``attention``.
        cfg: Optional config dict from YAML.
    """
    cfg = cfg or {}
    if method == "mean":
        return MeanFusion()
    if method == "weighted":
        return ConfidenceWeightedFusion(
            power=float(cfg.get("confidence_power", 1.0)),
            temperature=float(cfg.get("temperature", 1.0)),
        )
    if method == "attention":
        return AttentionFusionStrategy(
            hidden_channels=int(cfg.get("hidden_channels", 16)),
            num_heads=int(cfg.get("num_heads", 4)),
            dropout=float(cfg.get("dropout", 0.0)),
            temperature=float(cfg.get("temperature", 1.0)),
            use_flow_confidence=bool(cfg.get("use_flow_confidence", True)),
            confidence_power=float(cfg.get("confidence_power", 1.0)),
        )
    raise ValueError(f"Unknown fusion method: {method}")
