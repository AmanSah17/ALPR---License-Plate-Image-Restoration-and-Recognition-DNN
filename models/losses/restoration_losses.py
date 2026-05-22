"""
Composite restoration loss: L1 + SSIM + LPIPS (Phase 5.1 / 7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RestorationLossConfig:
    """Weights for composite restoration loss."""

    l1_weight: float = 1.0
    ssim_weight: float = 0.5
    lpips_weight: float = 0.1
    ssim_window_size: int = 11


class _SSIMLoss(nn.Module):
    """Differentiable ``1 - SSIM`` loss on ``(B, C, H, W)`` tensors in [0, 1]."""

    def __init__(self, window_size: int = 11, max_val: float = 1.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.c1 = (0.01 * max_val) ** 2
        self.c2 = (0.03 * max_val) ** 2

    def _gaussian_window(self, channel: int, dtype, device) -> torch.Tensor:
        size = self.window_size
        sigma = 1.5
        coords = torch.arange(size, dtype=dtype, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        window = (g @ g.t()).unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, size, size).contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        window = self._gaussian_window(c, pred.dtype, pred.device)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad, groups=c)
        mu_t = F.conv2d(target, window, padding=pad, groups=c)
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sig_p = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_p2
        sig_t = F.conv2d(target * target, window, padding=pad, groups=c) - mu_t2
        sig_pt = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_pt
        ssim = ((2 * mu_pt + self.c1) * (2 * sig_pt + self.c2)) / (
            (mu_p2 + mu_t2 + self.c1) * (sig_p + sig_t + self.c2)
        )
        return 1.0 - ssim.mean()


class RestorationLoss(nn.Module):
    """
    Weighted L1 + SSIM + LPIPS for training SwinIRUNetHybrid.

    Returns total loss and component dict for MLflow logging.
    """

    def __init__(self, config: Optional[RestorationLossConfig] = None) -> None:
        super().__init__()
        self.config = config or RestorationLossConfig()
        self.ssim_loss = _SSIMLoss(window_size=self.config.ssim_window_size)
        self._lpips: Optional[nn.Module] = None

    def _get_lpips(self, device: torch.device) -> nn.Module:
        if self._lpips is None:
            try:
                import lpips

                self._lpips = lpips.LPIPS(net="alex").to(device)
                for param in self._lpips.parameters():
                    param.requires_grad = False
                self._lpips.eval()
            except ImportError as exc:
                raise ImportError("Install lpips: pip install lpips") from exc
        return self._lpips

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute composite loss.

        Args:
            prediction: ``(B, 3, H, W)`` in [0, 1].
            target: Same shape as prediction.

        Returns:
            Total loss scalar and dict of components.
        """
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")

        l1 = F.l1_loss(prediction, target)
        ssim = self.ssim_loss(prediction, target)
        components: Dict[str, torch.Tensor] = {"l1": l1, "ssim": ssim}

        total = self.config.l1_weight * l1 + self.config.ssim_weight * ssim

        if self.config.lpips_weight > 0:
            lpips_model = self._get_lpips(prediction.device)
            min_spatial = min(prediction.shape[-2], prediction.shape[-1])
            if min_spatial < 32:
                new_h = max(prediction.shape[-2], 32)
                new_w = max(prediction.shape[-1], 32)
                is_3d = prediction.ndim == 3
                if is_3d:
                    pred_interp = F.interpolate(prediction.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                    tgt_interp = F.interpolate(target.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                else:
                    pred_interp = F.interpolate(prediction, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    tgt_interp = F.interpolate(target, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                pred_interp = prediction
                tgt_interp = target

            pred_n = pred_interp * 2 - 1
            tgt_n = tgt_interp * 2 - 1
            lp = lpips_model(pred_n, tgt_n).mean()
            components["lpips"] = lp
            total = total + self.config.lpips_weight * lp
        else:
            components["lpips"] = torch.zeros((), device=prediction.device)

        components["total"] = total
        return total, components

    @classmethod
    def from_config(cls, cfg: Any) -> "RestorationLoss":
        section = cfg.losses if hasattr(cfg, "losses") else cfg
        return cls(
            RestorationLossConfig(
                l1_weight=float(section.get("l1_weight", 1.0)),
                ssim_weight=float(section.get("ssim_weight", 0.5)),
                lpips_weight=float(section.get("lpips_weight", 0.1)),
            )
        )
