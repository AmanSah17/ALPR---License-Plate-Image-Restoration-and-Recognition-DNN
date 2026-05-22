"""
Restoration quality visualizations for Phase 5 reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import torch

PathLike = Union[str, Path]


class RestorationVisualizer:
    """Generate restoration metric charts and error comparisons."""

    def __init__(self, output_dir: PathLike, dpi: int = 150) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def save_metrics_bar(
        self,
        metrics: Dict[str, float],
        filename: str = "restoration_metrics.png",
    ) -> Path:
        """Bar chart of PSNR/SSIM restored vs baseline."""
        labels = ["PSNR restored", "PSNR baseline", "SSIM restored", "SSIM baseline"]
        keys = [
            "psnr_restored",
            "psnr_baseline_center",
            "ssim_restored",
            "ssim_baseline_center",
        ]
        values = [metrics.get(k, 0.0) for k in keys]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(labels, values, color=["seagreen", "gray", "steelblue", "lightgray"])
        ax.set_title("Restoration vs center-frame baseline")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def save_gain_summary(
        self,
        restored: torch.Tensor,
        baseline: torch.Tensor,
        target: torch.Tensor,
        metrics: Dict[str, float],
        filename: str,
    ) -> Path:
        """Annotated comparison with PSNR/SSIM gains."""
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        titles = [
            f"GT ROI",
            f"Center\nPSNR {metrics.get('psnr_baseline_center', 0):.2f}",
            f"Restored\nPSNR {metrics.get('psnr_restored', 0):.2f}",
        ]
        for ax, img, title in zip(axes, [target, baseline, restored], titles):
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        gain = metrics.get("psnr_gain_vs_center", 0)
        fig.suptitle(f"PSNR gain vs center: {gain:+.2f} dB", fontsize=10)
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out
