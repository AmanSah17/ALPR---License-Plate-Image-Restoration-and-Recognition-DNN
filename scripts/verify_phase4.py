#!/usr/bin/env python
"""
Phase-4 verification: temporal fusion (mean / weighted / attention) after Phase 3.

Uses gemma4::

    D:\\gemma4\\gemma4\\Scripts\\python.exe scripts\\verify_phase4.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env_paths import ensure_venv_on_path, get_python_executable

ensure_venv_on_path()

from datasets.rlpr_dataset import RLPRDataset
from engine.optical_flow_engine import OpticalFlowEngine
from engine.temporal_fusion_engine import TemporalFusionEngine
from logging_utils.experiment_manager import ExperimentManager
from logging_utils.phase_outputs import PhaseOutputManager
from models.optical_flow.flow_utils import numpy_hwc_to_chw_float
from utils.config_loader import ConfigManager
from utils.gpu_utils import GPUManager
from utils.seed_utils import set_seed
from visualization.fusion_visualizer import FusionVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase-4 temporal fusion.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=8, help="Phase-3 flow pairs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(42)

    cfg = ConfigManager(PROJECT_ROOT).loader.load("temporal_fusion.yaml")
    log_cfg = ConfigManager(PROJECT_ROOT).load("logging", validate=True)
    cfg.optical_flow.max_frame_pairs = args.max_pairs

    print(f"[INFO] Python: {get_python_executable()}")
    gpu = GPUManager.from_config(cfg.hardware)
    gpu.log_status()

    dataset = RLPRDataset.from_config(cfg, project_root=PROJECT_ROOT, training=False)
    sample = dataset.load_sample_raw(args.sample_index)
    ref_idx = sample.center_frame_index
    reference = numpy_hwc_to_chw_float(sample.frames[ref_idx])

    manager = ExperimentManager(PROJECT_ROOT, logging_cfg=log_cfg)
    tracker = manager.create(
        name="phase4_temporal_fusion",
        tags=["phase4", "temporal_fusion"],
        config_snapshot=cfg,
        seed=42,
    )

    phase3_paths = PhaseOutputManager(tracker.paths.root).get("optical_flow")
    phase4_paths = PhaseOutputManager(tracker.paths.root).get("temporal_fusion")
    print(f"[INFO] Phase-3 dir: {phase3_paths.root}")
    print(f"[INFO] Phase-4 dir: {phase4_paths.root}")

    try:
        # --- Phase 3 (dependency) ---
        flow_engine = OpticalFlowEngine.from_config(cfg, phase3_paths, gpu_manager=gpu)
        flow_result = flow_engine.run(sample.frames, sample.sample_id, ref_idx)
        print(
            f"[OK] Phase-3: pairs={len(flow_result.frame_indices)} "
            f"fps={flow_result.metrics['inference_fps']:.2f}"
        )

        # --- Phase 4 ---
        fusion_engine = TemporalFusionEngine.from_config(cfg, phase4_paths, gpu_manager=gpu)
        comparison = fusion_engine.compare(
            flow_result.warped,
            reference,
            flow_result.flows,
            sample.sample_id,
        )

        viz = FusionVisualizer(phase4_paths.visualizations)
        viz_paths = {
            "comparison": str(
                viz.save_method_comparison(
                    reference,
                    comparison.results,
                    f"{sample.sample_id}_method_comparison.png",
                )
            ),
        }
        if "attention" in comparison.results:
            attn = comparison.results["attention"]
            spatial = attn.extras.get("spatial_attention")
            if spatial is not None:
                viz_paths["attention"] = str(
                    viz.save_attention_heatmap(
                        spatial, f"{sample.sample_id}_attention_spatial.png"
                    )
                )
            viz_paths["weights"] = str(
                viz.save_weight_bar(
                    attn.weights,
                    flow_result.frame_indices,
                    "attention",
                    f"{sample.sample_id}_attention_weights.png",
                )
            )

        # MLflow metrics per method + phase3 context
        mlflow_metrics = {
            "phase3_flow_magnitude_mean": flow_result.metrics["flow_magnitude_mean"],
            "phase3_warp_l1_error": flow_result.metrics["warp_l1_error_mean"],
            "phase3_inference_fps": flow_result.metrics["inference_fps"],
            "fusion_best_method_score": comparison.metrics[comparison.best_method][
                "composite_score"
            ],
            "fusion_best_alignment_mse": comparison.metrics[comparison.best_method][
                "alignment_mse"
            ],
            "fusion_best_sharpness": comparison.metrics[comparison.best_method][
                "sharpness_laplacian"
            ],
        }
        for method, m in comparison.metrics.items():
            for k, v in m.items():
                mlflow_metrics[f"fusion/{method}/{k}"] = v
        mlflow_metrics["fusion/best_method_index"] = float(
            ["mean", "weighted", "attention"].index(comparison.best_method)
        )

        tracker.log_benchmark_metrics(mlflow_metrics, step=0)
        tracker.save_metrics_json(mlflow_metrics, "temporal_fusion_metrics.json", step=0)

        report = {
            "sample_id": sample.sample_id,
            "best_method": comparison.best_method,
            "phase3_metrics": flow_result.metrics,
            "fusion_metrics": comparison.metrics,
            "phase4_dirs": phase4_paths.as_dict(),
            "visualizations": viz_paths,
        }
        report_path = phase4_paths.metrics / "phase4_verify_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tracker.mlflow.log_artifact(report_path, artifact_path="temporal_fusion")

        print("\n[OK] Phase-4 fusion comparison")
        print(f"     Best method: {comparison.best_method}")
        for method in comparison.metrics:
            m = comparison.metrics[method]
            print(
                f"     {method:10s} mse={m['alignment_mse']:.5f} "
                f"sharp={m['sharpness_laplacian']:.2f} score={m['composite_score']:.4f}"
            )
        print(f"     Outputs: {phase4_paths.root}")
        print(f"     MLflow:  {tracker.mlflow.run_id}")
    finally:
        tracker.close()

    print("\n=== Phase-4 verification PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
