"""
Temporal smoothing of optical flow sequences (median / gaussian / EMA).
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FilterMethod = Literal["median", "gaussian", "ema"]


class TemporalFlowFilter:
    """
    Apply temporal filtering along the frame dimension of a flow stack.

    Input flows are warps **toward the reference frame** ordered by source frame index.
    """

    def __init__(
        self,
        method: FilterMethod = "median",
        window_size: int = 3,
        ema_alpha: float = 0.6,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            method: ``median``, ``gaussian``, or ``ema``.
            window_size: Odd window for median/gaussian.
            ema_alpha: EMA weight for newest frame (only for ``ema``).
            enabled: Pass-through when False.
        """
        self.method = method
        self.window_size = max(1, window_size | 1)  # force odd
        self.ema_alpha = ema_alpha
        self.enabled = enabled

    def filter(self, flows: torch.Tensor) -> torch.Tensor:
        """
        Filter flow tensor along temporal dimension.

        Args:
            flows: ``(T, 2, H, W)``.

        Returns:
            Filtered flows, same shape.
        """
        if not self.enabled or flows.shape[0] < 2:
            return flows

        if self.method == "median":
            return self._median_filter(flows)
        if self.method == "gaussian":
            return self._gaussian_filter(flows)
        if self.method == "ema":
            return self._ema_filter(flows)
        raise ValueError(f"Unknown filter method: {self.method}")

    def _median_filter(self, flows: torch.Tensor) -> torch.Tensor:
        """Per-pixel median over temporal window (memory-efficient loop)."""
        t, c, h, w = flows.shape
        half = self.window_size // 2
        out = flows.clone()
        for i in range(t):
            start = max(0, i - half)
            end = min(t, i + half + 1)
            window = flows[start:end]
            out[i] = torch.median(window, dim=0).values
        return out

    def _gaussian_filter(self, flows: torch.Tensor) -> torch.Tensor:
        """1D Gaussian smoothing over time (separable per channel)."""
        t = flows.shape[0]
        sigma = max(self.window_size / 6.0, 0.5)
        coords = torch.arange(t, dtype=torch.float32, device=flows.device) - (t - 1) / 2
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()
        # Conv1d over time: reshape (1, C*H*W, T)
        flat = flows.permute(1, 2, 3, 0).reshape(1, -1, t)
        pad = self.window_size // 2
        filtered = F.conv1d(
            F.pad(flat, (pad, pad), mode="replicate"),
            kernel.view(1, 1, -1).to(flows.device),
        )
        filtered = filtered.view(flows.shape[1], flows.shape[2], flows.shape[3], t)
        return filtered.permute(3, 0, 1, 2).contiguous()

    def _ema_filter(self, flows: torch.Tensor) -> torch.Tensor:
        """Exponential moving average along time."""
        out = flows.clone()
        acc = flows[0].clone()
        for i in range(1, flows.shape[0]):
            acc = self.ema_alpha * flows[i] + (1.0 - self.ema_alpha) * acc
            out[i] = acc
        return out

    @classmethod
    def from_config(cls, cfg: Any) -> "TemporalFlowFilter":
        """Build from ``optical_flow`` or ``optical_flow.temporal_filter`` config."""
        if hasattr(cfg, "temporal_filter"):
            tf = cfg.temporal_filter
        elif hasattr(cfg, "get"):
            tf = cfg.get("temporal_filter", {})
        else:
            tf = {}
        return cls(
            method=str(tf.get("method", "median")),
            window_size=int(tf.get("window_size", 3)),
            enabled=bool(tf.get("enabled", True)),
        )
