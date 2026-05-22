"""
RLPR dataset visualizations: frame grids, statistics, motion preview, GT comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from datasets.rlpr_dataset import RLPRDataset, RLPRSample
from utils.image_utils import save_image_rgb

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


class RLPRDatasetVisualizer:
    """
    Generate and save RLPR diagnostic figures for experiment artifacts.
    """

    def __init__(self, output_dir: PathLike, dpi: int = 150) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def save_frame_grid(
        self,
        sample: RLPRSample,
        filename: str = "frame_grid.png",
        cols: int = 8,
    ) -> Path:
        """
        Plot all frames in a grid with center frame highlighted.

        Args:
            sample: Raw RLPR sample.
            filename: Output filename.
            cols: Grid columns.

        Returns:
            Path to saved figure.
        """
        frames = sample.frames
        t = frames.shape[0]
        rows = int(np.ceil(t / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 0.8))
        axes_flat = np.array(axes).reshape(-1)

        for i in range(rows * cols):
            ax = axes_flat[i]
            ax.axis("off")
            if i < t:
                ax.imshow(frames[i])
                title = f"{i + 1:02d}"
                if i == sample.center_frame_index:
                    ax.set_title(title, color="lime", fontsize=8)
                else:
                    ax.set_title(title, fontsize=7)
            else:
                ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))

        fig.suptitle(f"{sample.sample_id} | {sample.plate_text_compact}", fontsize=10)
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved frame grid: %s", out)
        return out

    def save_gt_comparison(
        self,
        sample: RLPRSample,
        filename: str = "gt_vs_center.png",
    ) -> Path:
        """Side-by-side center frame vs pseudo-GT ROI vs SR ROI."""
        center = sample.frames[sample.center_frame_index]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        titles = ["Center LQ (16.png)", "Pseudo_GT_ROI", "SR_ROI (release)"]
        images = [center, sample.pseudo_gt_roi, sample.sr_roi]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{sample.sample_id} — {sample.plate_text_compact}")
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def save_motion_preview(
        self,
        sample: RLPRSample,
        filename: str = "motion_preview.png",
    ) -> Path:
        """
        Temporal difference magnitude between consecutive frames (motion proxy).
        """
        frames = sample.frames.astype(np.float32)
        # Collapse spatial dims -> (num_pairs, H) for imshow
        diffs = [
            np.mean(np.abs(frames[i + 1] - frames[i]), axis=2).mean(axis=1)
            for i in range(len(frames) - 1)
        ]
        stack = np.stack(diffs, axis=0)  # (T-1, H)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].imshow(frames[sample.center_frame_index])
        axes[0].set_title("Center frame")
        axes[0].axis("off")

        im = axes[1].imshow(stack, aspect="auto", cmap="magma")
        axes[1].set_title("Frame-to-frame mean |diff|")
        axes[1].set_xlabel("Width")
        axes[1].set_ylabel("Frame pair index")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
        fig.suptitle(f"Motion preview — {sample.sample_id}")
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def save_frame_statistics(
        self,
        dataset: RLPRDataset,
        max_samples: Optional[int] = None,
        filename: str = "frame_statistics.png",
    ) -> Path:
        """
        Plot distributions of center-frame width/height across the dataset.
        """
        n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        widths, heights, lengths = [], [], []

        for i in range(n):
            sample = dataset.load_sample_raw(i)
            h, w, _ = sample.frames[sample.center_frame_index].shape
            widths.append(w)
            heights.append(h)
            lengths.append(len(sample.plate_text_compact))

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].hist(widths, bins=20, color="steelblue", edgecolor="white")
        axes[0].set_title("Center frame width")
        axes[1].hist(heights, bins=20, color="coral", edgecolor="white")
        axes[1].set_title("Center frame height")
        axes[2].hist(lengths, bins=range(5, 10), color="seagreen", edgecolor="white", align="left")
        axes[2].set_title("Label length (compact)")
        fig.suptitle(f"RLPR statistics (n={n})")
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def generate_sample_report(
        self,
        sample: RLPRSample,
        prefix: str,
    ) -> Dict[str, str]:
        """Generate all per-sample visualizations; return path mapping."""
        return {
            "frame_grid": str(self.save_frame_grid(sample, f"{prefix}_frame_grid.png")),
            "gt_comparison": str(self.save_gt_comparison(sample, f"{prefix}_gt_vs_center.png")),
            "motion_preview": str(self.save_motion_preview(sample, f"{prefix}_motion.png")),
        }
