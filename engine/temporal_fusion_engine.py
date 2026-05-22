"""
Temporal fusion engine: compare mean / weighted / attention and pick best by metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from logging_utils.phase_outputs import PhaseOutputPaths
from models.restoration.fusion_strategies import (
    BaseFusionStrategy,
    FusionMethod,
    FusionResult,
    build_fusion_strategy,
)
from utils.gpu_utils import GPUManager
from utils.image_utils import save_image_rgb
from utils.timing import BenchmarkTimer

logger = logging.getLogger(__name__)


@dataclass
class FusionComparisonResult:
    """Results for all fusion methods on one sample."""

    sample_id: str
    results: Dict[str, FusionResult]
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_method: str = ""
    artifact_paths: Dict[str, str] = field(default_factory=dict)


class TemporalFusionEngine:
    """
    Apply and compare temporal fusion strategies on warped frame stacks.

    Metrics (lower is better unless noted):
        - ``alignment_mse``: MSE vs center reference (lower = better alignment).
        - ``temporal_variance``: Pixel variance across frames (lower = more stable).
        - ``sharpness_laplacian``: Variance of Laplacian on fused image (higher = sharper).
        - ``weight_entropy``: Frame weight entropy (lower = more selective).
    """

    def __init__(
        self,
        methods: List[FusionMethod],
        output_paths: PhaseOutputPaths,
        fusion_cfg: Optional[dict] = None,
        primary_method: str = "attention",
        save_fused_png: bool = True,
        gpu_manager: Optional[GPUManager] = None,
    ) -> None:
        self.methods = methods
        self.outputs = output_paths
        self.fusion_cfg = fusion_cfg or {}
        self.primary_method = primary_method
        self.save_fused_png = save_fused_png
        self.gpu = gpu_manager or GPUManager()
        self._strategies: Dict[str, BaseFusionStrategy] = {
            m: build_fusion_strategy(m, self.fusion_cfg) for m in methods
        }

    @staticmethod
    def _alignment_mse(fused: torch.Tensor, reference: torch.Tensor) -> float:
        return float(torch.mean((fused - reference) ** 2).item())

    @staticmethod
    def _temporal_variance(frames: torch.Tensor) -> float:
        return float(torch.var(frames, dim=0).mean().item())

    @staticmethod
    def _sharpness_laplacian(image: torch.Tensor) -> float:
        """Laplacian variance sharpness proxy on ``(3,H,W)`` in [0,1]."""
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
        kernel = torch.tensor(
            [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
            dtype=gray.dtype,
            device=gray.device,
        )
        lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
        return float(lap.var().item())

    @staticmethod
    def _weight_entropy(weights: torch.Tensor) -> float:
        w = weights.clamp(min=1e-8)
        return float((-(w * torch.log(w))).sum().item())

    def _score_method(self, metrics: Dict[str, float]) -> float:
        """
        Higher is better composite score for method selection.

        Rewards low alignment error, high sharpness, moderate entropy.
        """
        return (
            -metrics["alignment_mse"] * 10.0
            + metrics["sharpness_laplacian"] * 0.01
            - metrics.get("temporal_variance_warped", 0.0) * 0.1
        )

    def compare(
        self,
        warped: torch.Tensor,
        reference: torch.Tensor,
        flows: Optional[torch.Tensor],
        sample_id: str,
    ) -> FusionComparisonResult:
        """
        Run all configured fusion methods and compute metrics.

        Args:
            warped: ``(T, 3, H, W)`` from Phase 3.
            reference: ``(3, H, W)`` center frame [0,1].
            flows: Optional ``(T, 2, H, W)``.
            sample_id: For artifact naming.

        Returns:
            ``FusionComparisonResult`` with best method by composite score.
        """
        ref_var = self._temporal_variance(warped)
        fusion_results: Dict[str, FusionResult] = {}
        all_metrics: Dict[str, Dict[str, float]] = {}

        timer = BenchmarkTimer(name="fusion", warmup=0)
        for method in self.methods:
            strategy = self._strategies[method]
            with timer.measure():
                result = strategy.fuse(warped, reference, flows)
            m = {
                "alignment_mse": self._alignment_mse(result.fused, reference),
                "temporal_variance_warped": ref_var,
                "fused_pixel_variance": float(torch.var(result.fused).item()),
                "sharpness_laplacian": self._sharpness_laplacian(result.fused),
                "weight_entropy": self._weight_entropy(result.weights),
                "fusion_time_ms": timer.samples[-1] if timer.samples else 0.0,
                "composite_score": 0.0,
            }
            m["composite_score"] = self._score_method(m)
            fusion_results[method] = result
            all_metrics[method] = m
            logger.info(
                "[%s] %s: mse=%.5f sharp=%.3f score=%.4f",
                sample_id,
                method,
                m["alignment_mse"],
                m["sharpness_laplacian"],
                m["composite_score"],
            )

        best = max(all_metrics.items(), key=lambda x: x[1]["composite_score"])[0]
        paths = self._save_artifacts(sample_id, fusion_results, all_metrics, best)

        return FusionComparisonResult(
            sample_id=sample_id,
            results=fusion_results,
            metrics=all_metrics,
            best_method=best,
            artifact_paths=paths,
        )

    def _save_artifacts(
        self,
        sample_id: str,
        results: Dict[str, FusionResult],
        metrics: Dict[str, Dict[str, float]],
        best_method: str,
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        fused_dir = self.outputs.root / "fused"
        weights_dir = self.outputs.root / "weights"
        fused_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)

        for method, result in results.items():
            if self.save_fused_png:
                img = (
                    result.fused.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255
                ).astype("uint8")
                img = img.astype(np.uint8)
                p = fused_dir / f"{sample_id}_{method}_fused.png"
                save_image_rgb(img, p)
                paths[f"fused_{method}"] = str(p)

            w_path = weights_dir / f"{sample_id}_{method}_weights.json"
            w_path.write_text(
                json.dumps(
                    {
                        "weights": result.weights.detach().cpu().tolist(),
                        "metrics": metrics[method],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            paths[f"weights_{method}"] = str(w_path)

        metrics_path = self.outputs.metrics / f"{sample_id}_fusion_comparison.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"best_method": best_method, "metrics": metrics},
                handle,
                indent=2,
            )
        paths["comparison_json"] = str(metrics_path)
        return paths

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        output_paths: PhaseOutputPaths,
        gpu_manager: Optional[GPUManager] = None,
    ) -> "TemporalFusionEngine":
        tf = cfg.temporal_fusion
        methods = list(tf.get("methods", ["mean", "weighted", "attention"]))
        return cls(
            methods=methods,  # type: ignore[arg-type]
            output_paths=output_paths,
            fusion_cfg=dict(tf),
            primary_method=str(tf.get("primary_method", "attention")),
            save_fused_png=bool(cfg.output.get("save_fused_png", True)),
            gpu_manager=gpu_manager,
        )
