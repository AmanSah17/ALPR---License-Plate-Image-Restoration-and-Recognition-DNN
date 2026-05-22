#!/usr/bin/env python
"""
Phase-3 verification: RAFT-small optical flow, temporal filter, warping, MLflow.

Uses gemma4 environment::

    D:\\gemma4\\gemma4\\Scripts\\python.exe scripts\\verify_phase3.py
    .\\scripts\\run.ps1 scripts\\verify_phase3.py
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
from logging_utils.experiment_manager import ExperimentManager
from logging_utils.phase_outputs import PhaseOutputManager
from utils.config_loader import ConfigManager
from utils.gpu_utils import GPUManager
from utils.seed_utils import set_seed
from models.optical_flow.flow_utils import numpy_hwc_to_chw_float
from visualization.flow_visualizer import FlowVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase-3 optical flow pipeline.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=8,
        help="Max frame pairs to process (VRAM/time control).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(42)

    cfg = ConfigManager(PROJECT_ROOT).loader.load("optical_flow.yaml")
    log_cfg = ConfigManager(PROJECT_ROOT).load("logging", validate=True)

    # Override for faster verification
    cfg.optical_flow.max_frame_pairs = args.max_pairs

    print(f"[INFO] Python: {get_python_executable()}")
    gpu = GPUManager.from_config(cfg.hardware)
    gpu.log_status()

    dataset = RLPRDataset.from_config(cfg, project_root=PROJECT_ROOT, training=False)
    sample = dataset.load_sample_raw(args.sample_index)
    ref_idx = sample.center_frame_index

    manager = ExperimentManager(PROJECT_ROOT, logging_cfg=log_cfg)
    tracker = manager.create(
        name="phase3_optical_flow",
        tags=["phase3", "optical_flow", "raft"],
        config_snapshot=cfg,
        seed=42,
    )

    phase_paths = PhaseOutputManager(tracker.paths.root).get("optical_flow")
    print(f"[INFO] Phase-3 outputs: {phase_paths.root}")

    try:
        engine = OpticalFlowEngine.from_config(cfg, phase_paths, gpu_manager=gpu)
        result = engine.run(sample.frames, sample.sample_id, ref_idx)

        # Visualizations in dedicated phase folder
        viz = FlowVisualizer(phase_paths.visualizations)
        mid = len(result.frame_indices) // 2
        mid_idx = result.frame_indices[mid]
        tensors = [numpy_hwc_to_chw_float(sample.frames[i]) for i in range(sample.frames.shape[0])]
        ref_tensor = tensors[ref_idx]
        src_tensor = tensors[mid_idx]
        flow_mid = result.flows[mid]
        warped_mid = result.warped[mid]

        viz_paths = {
            "hsv": str(viz.save_hsv(flow_mid, f"{sample.sample_id}_flow_hsv.png")),
            "magnitude": str(
                viz.save_magnitude_heatmap(flow_mid, f"{sample.sample_id}_flow_mag.png")
            ),
            "consistency": str(
                viz.save_temporal_consistency_plot(
                    result.flows, f"{sample.sample_id}_temporal_consistency.png"
                )
            ),
            "panel": str(
                viz.save_comparison_panel(
                    ref_tensor,
                    src_tensor,
                    warped_mid,
                    flow_mid,
                    f"{sample.sample_id}_comparison.png",
                )
            ),
        }

        # Benchmark dedicated RAFT pair timing
        bench = engine.benchmark_pair_inference(
            src_tensor,
            ref_tensor,
            warmup=int(cfg.benchmark.warmup_iterations),
            iterations=int(cfg.benchmark.timed_iterations),
        )

        mem = gpu.memory_stats()
        all_metrics = {
            **result.metrics,
            **{f"bench_{k}": v for k, v in bench.items()},
            "gpu_peak_allocated_mb": mem.max_allocated_mb,
            "gpu_allocated_mb": mem.allocated_mb,
        }

        tracker.log_benchmark_metrics(all_metrics, step=0)
        tracker.save_metrics_json(all_metrics, "optical_flow_metrics.json", step=0)

        report = {
            "sample_id": sample.sample_id,
            "phase_output_dirs": phase_paths.as_dict(),
            "metrics": all_metrics,
            "artifacts": result.artifact_paths,
            "visualizations": viz_paths,
        }
        report_path = phase_paths.metrics / "phase3_verify_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tracker.paths.reports.mkdir(parents=True, exist_ok=True)
        summary = tracker.paths.reports / "phase3_verify_summary.json"
        summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

        for path in list(result.artifact_paths.values()) + list(viz_paths.values()):
            tracker.mlflow.log_artifact(path, artifact_path="optical_flow")
        tracker.mlflow.log_artifact(report_path, artifact_path="optical_flow/metrics")

        print("[OK] Optical flow pipeline complete.")
        print(f"     Flow magnitude mean: {result.metrics['flow_magnitude_mean']:.4f}")
        print(f"     Warp L1 error mean:  {result.metrics['warp_l1_error_mean']:.4f}")
        print(f"     Inference FPS:       {result.metrics['inference_fps']:.2f}")
        print(f"     Artifacts:           {phase_paths.root}")
        print(f"     MLflow run_id:       {tracker.mlflow.run_id}")
    finally:
        tracker.close()

    print("\n=== Phase-3 verification PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
