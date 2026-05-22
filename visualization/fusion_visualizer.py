"""
Temporal fusion visualizations: attention weights, method comparison panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.restoration.fusion_strategies import FusionResult

PathLike = Union[str, Path]


class FusionVisualizer:
    """Save fusion diagnostics into phase visualization folder."""

    def __init__(self, output_dir: PathLike, dpi: int = 150) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def save_weight_bar(
        self,
        weights: torch.Tensor,
        frame_indices: List[int],
        method: str,
        filename: str,
    ) -> Path:
        """Bar chart of per-frame fusion weights."""
        w = weights.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 3))
        labels = [f"f{i+1:02d}" for i in frame_indices]
        ax.bar(range(len(w)), w, color="steelblue", edgecolor="white")
        ax.set_xticks(range(len(w)))
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel("Weight")
        ax.set_title(f"Frame importance — {method}")
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def save_method_comparison(
        self,
        reference: torch.Tensor,
        results: Dict[str, FusionResult],
        filename: str = "method_comparison.png",
    ) -> Path:
        """Grid comparing fused outputs from each method vs reference."""
        methods = list(results.keys())
        n = len(methods) + 1
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]

        ref_np = _chw_to_hwc(reference)
        axes[0].imshow(ref_np)
        axes[0].set_title("Reference")
        axes[0].axis("off")

        for ax, method in zip(axes[1:], methods):
            img = _chw_to_hwc(results[method].fused)
            ax.imshow(img)
            ax.set_title(method)
            ax.axis("off")

        fig.suptitle("Fusion method comparison")
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def save_attention_heatmap(
        self,
        spatial_maps: torch.Tensor,
        filename: str,
    ) -> Path:
        """Mean spatial gate across frames."""
        if spatial_maps is None or spatial_maps.numel() == 0:
            raise ValueError("No spatial attention maps to visualize.")
        mean_map = spatial_maps.mean(dim=0)[0].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.imshow(mean_map, cmap="hot")
        ax.set_title("Mean spatial attention")
        ax.axis("off")
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out


def _chw_to_hwc(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
