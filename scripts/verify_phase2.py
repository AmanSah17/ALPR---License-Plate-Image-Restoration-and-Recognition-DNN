#!/usr/bin/env python
"""
Phase-2 verification: RLPR dataset load, validation, visualization, MLflow metrics.

Uses gemma4 environment (see scripts/activate_env.ps1).

    D:\\gemma4\\gemma4\\Scripts\\python.exe scripts/verify_phase2.py
    .\\scripts\\run.ps1 scripts\\verify_phase2.py
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
from datasets.validators import RLPRDatasetValidator
from logging_utils.experiment_manager import ExperimentManager
from torch.utils.data import DataLoader
from utils.config_loader import ConfigManager
from visualization.dataset_visualizer import RLPRDatasetVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase-2 RLPR dataset pipeline.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ConfigManager(PROJECT_ROOT).load("dataset", validate=True)
    log_cfg = ConfigManager(PROJECT_ROOT).load("logging", validate=True)

    root = PROJECT_ROOT / cfg.dataset.root_dir
    print(f"[INFO] RLPR root: {root}")
    print(f"[INFO] Python: {get_python_executable()}")

    report = RLPRDatasetValidator(
        root,
        num_frames=int(cfg.dataset.num_frames),
        center_frame_name=f"{int(cfg.dataset.center_frame_index):02d}.png",
        strict_paths=bool(cfg.validation.strict_paths),
    ).validate()
    print(f"[OK] Validation: {report.sample_count} samples, valid={report.valid}")

    dataset = RLPRDataset.from_config(cfg, project_root=PROJECT_ROOT, training=False)
    sample = dataset.load_sample_raw(args.sample_index)
    print(
        f"[OK] Loaded {sample.sample_id}: frames={sample.frames.shape}, "
        f"label={sample.plate_text_compact}"
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.loading.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(cfg.loading.pin_memory),
    )
    batch = next(iter(loader))
    print(
        f"[OK] DataLoader batch: frames={tuple(batch['frames'].shape)}, "
        f"center={tuple(batch['center_frame'].shape)}, "
        f"gt={tuple(batch['pseudo_gt_roi'].shape)}"
    )

    manager = ExperimentManager(PROJECT_ROOT, logging_cfg=log_cfg)
    tracker = manager.create(
        name="phase2_verify",
        tags=["phase2", "rlpr", "dataset"],
        config_snapshot=cfg,
        seed=42,
    )

    try:
        viz = RLPRDatasetVisualizer(tracker.paths.visualizations)
        paths = viz.generate_sample_report(sample, prefix=sample.sample_id)
        stats_path = viz.save_frame_statistics(dataset, max_samples=50)

        for key, path in {**paths, "statistics": str(stats_path)}.items():
            tracker.artifacts.register(
                Path(path),
                name=f"phase2_{key}",
                category="visualizations",
                copy_file=False,
            )
            tracker.mlflow.log_artifact(path, artifact_path="visualizations")

        # Accurate MLflow dataset metrics (integer counts + float stats)
        center = sample.frames[sample.center_frame_index]
        h, w, _ = center.shape
        tracker.log_benchmark_metrics(
            {
                "dataset/num_samples": float(len(dataset)),
                "dataset/num_frames_per_sample": float(sample.frames.shape[0]),
                "dataset/center_width": float(w),
                "dataset/center_height": float(h),
                "dataset/label_length": float(len(sample.plate_text_compact)),
                "dataset/validation_passed": 1.0 if report.valid else 0.0,
            },
            step=0,
        )

        summary = {
            "validation": report.to_dict(),
            "sample": sample.to_dict(),
            "visualizations": paths,
            "batch_shapes": {
                "frames": list(batch["frames"].shape),
                "center_frame": list(batch["center_frame"].shape),
                "pseudo_gt_roi": list(batch["pseudo_gt_roi"].shape),
            },
        }
        summary_path = tracker.paths.reports / "phase2_verify_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tracker.mlflow.log_artifact(summary_path, artifact_path="reports")
        tracker.save_metrics_json(summary["validation"], "validation.json", step=0)

        print("[OK] Visualizations saved and logged to MLflow.")
        print(f"Experiment: {tracker.paths.root}")
        print(f"MLflow run_id: {tracker.mlflow.run_id}")
    finally:
        tracker.close()

    print("\n=== Phase-2 verification PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
