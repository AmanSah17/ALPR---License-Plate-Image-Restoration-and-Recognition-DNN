"""
Optical flow visualizations: HSV encoding, magnitude heatmaps, temporal consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.optical_flow.flow_utils import FlowUtils

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def flow_to_hsv_rgb(flow: np.ndarray, max_magnitude: Optional[float] = None) -> np.ndarray:
    """
    Convert ``(2, H, W)`` flow to HSV color wheel RGB uint8.

    Args:
        flow: Optical flow numpy array.
        max_magnitude: Clip magnitude; auto if None.

    Returns:
        RGB uint8 ``(H, W, 3)``.
    """
    u = flow[0]
    v = flow[1]
    mag, ang = cv2_flow_polar(u, v)  # uses numpy below
    if max_magnitude is None:
        max_magnitude = max(float(mag.max()), 1e-6)
    mag = np.clip(mag / max_magnitude, 0, 1)
    hsv = np.zeros((*mag.shape, 3), dtype=np.float32)
    hsv[..., 0] = ang
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag
    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)


def cv2_flow_polar(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle and magnitude from flow components (no cv2 dependency)."""
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = (np.arctan2(v, u) + np.pi) / (2 * np.pi)  # [0,1]
    return mag, ang


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorized HSV [0,1] to RGB [0,1]."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    out = np.zeros((*h.shape, 3), dtype=np.float32)
    conditions = [
        (i == 0, np.stack([v, t, p], axis=-1)),
        (i == 1, np.stack([q, v, p], axis=-1)),
        (i == 2, np.stack([p, v, t], axis=-1)),
        (i == 3, np.stack([p, q, v], axis=-1)),
        (i == 4, np.stack([t, p, v], axis=-1)),
        (i == 5, np.stack([v, p, q], axis=-1)),
    ]
    for cond, val in conditions:
        out = np.where(cond[..., None], val, out)
    return np.clip(out, 0, 1)


class FlowVisualizer:
    """Save flow diagnostic figures into phase visualization folder."""

    def __init__(self, output_dir: PathLike, dpi: int = 150) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def save_hsv(self, flow: torch.Tensor, filename: str) -> Path:
        """HSV flow colorization."""
        flow_np = flow.detach().cpu().numpy()
        rgb = flow_to_hsv_rgb(flow_np)
        out = self.output_dir / filename
        plt.imsave(out, rgb)
        return out

    def save_magnitude_heatmap(self, flow: torch.Tensor, filename: str) -> Path:
        """Magnitude heatmap with colorbar."""
        mag = FlowUtils.flow_magnitude(flow.unsqueeze(0) if flow.dim() == 3 else flow)
        mag_np = mag.squeeze().detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 2))
        im = ax.imshow(mag_np, cmap="inferno")
        ax.set_title("Flow magnitude")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def save_temporal_consistency_plot(
        self,
        flows: torch.Tensor,
        filename: str = "temporal_consistency.png",
    ) -> Path:
        """
        Plot consecutive-flow L1 difference across the sequence.

        Args:
            flows: ``(T, 2, H, W)``.
        """
        diffs = []
        for i in range(flows.shape[0] - 1):
            d = torch.mean(torch.abs(flows[i + 1] - flows[i])).item()
            diffs.append(d)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(len(diffs)), diffs, marker="o")
        ax.set_xlabel("Pair index")
        ax.set_ylabel("Mean |Δflow|")
        ax.set_title("Temporal flow consistency")
        ax.grid(True, alpha=0.3)
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def save_comparison_panel(
        self,
        reference: torch.Tensor,
        source: torch.Tensor,
        warped: torch.Tensor,
        flow: torch.Tensor,
        filename: str,
    ) -> Path:
        """2x2 panel: source, reference, warped, HSV flow."""
        ref_np = _chw_to_hwc_uint8(reference)
        src_np = _chw_to_hwc_uint8(source)
        warped_np = _chw_to_hwc_uint8(warped)
        hsv = flow_to_hsv_rgb(flow.detach().cpu().numpy())

        fig, axes = plt.subplots(2, 2, figsize=(8, 4))
        axes[0, 0].imshow(src_np)
        axes[0, 0].set_title("Source")
        axes[0, 1].imshow(ref_np)
        axes[0, 1].set_title("Reference")
        axes[1, 0].imshow(warped_np)
        axes[1, 0].set_title("Warped")
        axes[1, 1].imshow(hsv)
        axes[1, 1].set_title("Flow HSV")
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out


def _chw_to_hwc_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
