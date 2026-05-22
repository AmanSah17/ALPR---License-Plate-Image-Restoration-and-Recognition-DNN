#!/usr/bin/env python
"""
Phase-1 verification: config loading, MLflow metrics, artifacts, GPU utilities.

Run from project root with gemma4 (CUDA + Numba) environment::

    . scripts/activate_env.ps1
    python scripts/verify_phase1.py --profile train

Or without activating::

    .\\scripts\\run.ps1 scripts\\verify_phase1.py --profile train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env_paths import ensure_venv_on_path, get_python_executable

ensure_venv_on_path()

from logging_utils.experiment_manager import ExperimentManager
from utils.config_loader import ConfigManager, ConfigError
from utils.gpu_utils import GPUManager
from utils.profiler import RuntimeProfiler, run_timed_iterations
from utils.seed_utils import set_seed
from utils.timing import BenchmarkTimer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase-1 infrastructure.")
    parser.add_argument(
        "--profile",
        type=str,
        default="logging",
        choices=["logging", "train", "inference", "model", "dataset"],
        help="Config profile to load and snapshot.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="phase1_verify",
        help="Short experiment name suffix.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def verify_config(profile: str) -> dict:
    """Load and validate a config profile."""
    manager = ConfigManager(PROJECT_ROOT)
    cfg = manager.load(profile, validate=profile in ConfigManager.REQUIRED_KEYS)
    resolved = manager.loader.to_json(cfg)
    print(f"[OK] Config profile '{profile}' loaded ({len(resolved)} chars).")
    return {"profile": profile, "config_chars": len(resolved)}


def verify_mlflow_metrics(tracker) -> dict:
    """
    Log synthetic train/val/benchmark metrics with explicit steps.

    MLflow should show monotonic steps and correct prefixes.
    """
    # Training loop simulation
    train_losses = []
    for epoch in range(3):
        loss = 1.0 / (epoch + 1)
        psnr = 20.0 + epoch * 1.5
        tracker.log_train_metrics({"loss": loss, "psnr": psnr}, step=epoch)
        tracker.log_val_metrics({"loss": loss * 1.1, "psnr": psnr - 0.5}, step=epoch)
        train_losses.append(loss)

    # Benchmark metrics (single step)
    timer = BenchmarkTimer(name="dummy_inference", warmup=2)
    stats = run_timed_iterations(lambda: None, timer, iterations=10, warmup=2)
    # Benchmark metrics use a dedicated step after training epochs (avoids MLflow step clash)
    bench_step = 3
    tracker.log_benchmark_metrics(
        {
            "fps": stats.fps,
            "latency_mean_ms": stats.mean_ms,
            "latency_p95_ms": stats.p95_ms,
        },
        step=bench_step,
    )

    # JSON metrics artifact
    metrics_path = tracker.save_metrics_json(
        {"final_train_loss": train_losses[-1], "epochs": 3},
        filename="verify_metrics.json",
        step=2,
    )

    print("[OK] MLflow metrics logged (train/, val/, benchmark/).")
    return {
        "epochs_logged": 3,
        "final_train_loss": train_losses[-1],
        "benchmark_fps": stats.fps,
        "metrics_json": str(metrics_path),
        "mlflow_run_id": tracker.mlflow.run_id,
    }


def verify_artifacts(tracker) -> dict:
    """Register a dummy report artifact."""
    report = tracker.paths.reports / "phase1_verify_report.json"
    report.write_text(json.dumps({"status": "ok"}, indent=2), encoding="utf-8")
    record = tracker.artifacts.register(
        report,
        name="phase1_report",
        category="reports",
        copy_file=False,
        metadata={"phase": 1},
    )
    tracker.mlflow.log_artifact(report, artifact_path="reports")
    print(f"[OK] Artifact registered: {record.path}")
    return {"artifact": record.to_dict()}


def verify_gpu() -> dict:
    """Report GPU memory and autocast context."""
    gpu = GPUManager.from_config({"hardware": {"device": "auto", "precision": "fp16"}})
    stats = gpu.memory_stats()
    gpu.log_status()
    print(f"[OK] Device={gpu.device} alloc={stats.allocated_mb:.1f}MB")
    return stats.to_dict()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    results: dict = {"checks": {}}
    try:
        results["checks"]["config"] = verify_config(args.profile)
    except ConfigError as exc:
        print(f"[FAIL] Config: {exc}")
        return 1

    results["checks"]["gpu"] = verify_gpu()

    # Full experiment with MLflow
    log_cfg = ConfigManager(PROJECT_ROOT).load("logging", validate=True)
    train_cfg = None
    if args.profile != "logging":
        train_cfg = ConfigManager(PROJECT_ROOT).load(args.profile, validate=False)

    manager = ExperimentManager(PROJECT_ROOT, logging_cfg=log_cfg)
    tracker = manager.create(
        name=args.experiment_name,
        tags=["phase1", "verify"],
        notes="Automated Phase-1 verification run",
        config_snapshot=train_cfg,
        seed=args.seed,
    )

    try:
        results["checks"]["mlflow"] = verify_mlflow_metrics(tracker)
        results["checks"]["artifacts"] = verify_artifacts(tracker)
    finally:
        tracker.close(status="FINISHED")

    summary_path = PROJECT_ROOT / "outputs" / "phase1_verify_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n=== Phase-1 verification PASSED ===")
    print(f"Python: {get_python_executable()}")
    print(f"MLflow run_id: {results['checks']['mlflow'].get('mlflow_run_id')}")
    print(f"Experiment root: {tracker.paths.root}")
    print(f"Summary: {summary_path}")
    print("\nView MLflow UI:")
    print(f"  mlflow ui --backend-store-uri file:///{tracker.paths.mlruns.resolve().as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
