#!/usr/bin/env python
"""
Phase-5 verification: SwinIR-UNet restoration after Phase 3+4 (gemma4 env).

    D:\\gemma4\\gemma4\\Scripts\\python.exe scripts\\verify_phase5.py --max-pairs 8
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
from engine.restoration_engine import RestorationEngine
from engine.temporal_fusion_engine import TemporalFusionEngine
from logging_utils.experiment_manager import ExperimentManager
from logging_utils.phase_outputs import PhaseOutputManager
from models.optical_flow.flow_utils import numpy_hwc_to_chw_float
from models.restoration.fusion_strategies import build_fusion_strategy
from models.restoration.swinir_unet import SwinIRUNetHybrid
from utils.config_loader import ConfigManager
from utils.gpu_utils import GPUManager
from utils.profiler import count_parameters, estimate_flops
from utils.seed_utils import set_seed
from visualization.restoration_visualizer import RestorationVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase-5 restoration pipeline.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument(
        "--fusion-method",
        type=str,
        default=None,
        help="Override fusion method (mean|weighted|attention).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Phase 5.1 trained checkpoint (.ckpt).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(42)

    cfg = ConfigManager(PROJECT_ROOT).loader.load("restoration.yaml")
    log_cfg = ConfigManager(PROJECT_ROOT).load("logging", validate=True)
    cfg.optical_flow.max_frame_pairs = args.max_pairs

    fusion_method = args.fusion_method or str(cfg.inference.get("fusion_method", "weighted"))

    print(f"[INFO] Python: {get_python_executable()}")
    gpu = GPUManager.from_config(cfg.hardware)
    gpu.log_status()

    dataset = RLPRDataset.from_config(cfg, project_root=PROJECT_ROOT, training=False)
    sample = dataset.load_sample_raw(args.sample_index)
    ref_idx = sample.center_frame_index
    reference = numpy_hwc_to_chw_float(sample.frames[ref_idx])
    pseudo_gt_t = numpy_hwc_to_chw_float(sample.pseudo_gt_roi)

    manager = ExperimentManager(PROJECT_ROOT, logging_cfg=log_cfg)
    tracker = manager.create(
        name="phase5_restoration",
        tags=["phase5", "restoration", "swinir_unet"],
        config_snapshot=cfg,
        seed=42,
    )

    p3 = PhaseOutputManager(tracker.paths.root).get("optical_flow")
    p4 = PhaseOutputManager(tracker.paths.root).get("temporal_fusion")
    p5 = PhaseOutputManager(tracker.paths.root).get("restoration")
    print(f"[INFO] Phase-5 outputs: {p5.root}")

    try:
        # Phase 3
        flow_engine = OpticalFlowEngine.from_config(cfg, p3, gpu_manager=gpu)
        flow_result = flow_engine.run(sample.frames, sample.sample_id, ref_idx)
        print(f"[OK] Phase-3 fps={flow_result.metrics['inference_fps']:.2f}")

        # Phase 4
        fusion_engine = TemporalFusionEngine.from_config(cfg, p4, gpu_manager=gpu)
        comparison = fusion_engine.compare(
            flow_result.warped, reference, flow_result.flows, sample.sample_id
        )
        fused_result = comparison.results[fusion_method]
        print(f"[OK] Phase-4 fusion={fusion_method} (best={comparison.best_method})")

        # Phase 5
        if args.checkpoint:
            cfg.restoration.checkpoint_path = args.checkpoint
        restore_engine = RestorationEngine.from_config(cfg, p5, gpu_manager=gpu)
        rest_result = restore_engine.run(
            fused_result.fused,
            reference,
            pseudo_gt_t,
            sample.sample_id,
        )

        # Architecture stats
        model = restore_engine.model
        n_params = count_parameters(model)
        flops = None
        try:
            dummy = torch.randn(1, 3, rest_result.target.shape[1], rest_result.target.shape[2])
            flops = estimate_flops(model.cpu(), dummy)
            model.to(restore_engine.gpu.device)
        except Exception:
            model.to(restore_engine.gpu.device)

        viz = RestorationVisualizer(p5.visualizations)
        viz_paths = {
            "metrics_bar": str(
                viz.save_metrics_bar(rest_result.metrics, f"{sample.sample_id}_metrics.png")
            ),
            "gain_panel": str(
                viz.save_gain_summary(
                    rest_result.restored,
                    rest_result.baseline_center,
                    rest_result.target,
                    rest_result.metrics,
                    f"{sample.sample_id}_gain_panel.png",
                )
            ),
        }

        mlflow_metrics = {
            **{f"phase3/{k}": v for k, v in flow_result.metrics.items()},
            **{f"fusion/{fusion_method}/{k}": v for k, v in comparison.metrics[fusion_method].items()},
            **{f"restoration/{k}": v for k, v in rest_result.metrics.items()},
            "model/num_parameters": float(n_params),
        }
        if flops is not None:
            mlflow_metrics["model/flops"] = float(flops)

        tracker.log_benchmark_metrics(mlflow_metrics, step=0)
        tracker.save_metrics_json(mlflow_metrics, "restoration_metrics.json", step=0)

        report = {
            "sample_id": sample.sample_id,
            "fusion_method": fusion_method,
            "plate_text": sample.plate_text_compact,
            "metrics": rest_result.metrics,
            "model_parameters": n_params,
            "flops": flops,
            "phase5_dirs": p5.as_dict(),
            "visualizations": viz_paths,
        }
        report_path = p5.metrics / "phase5_verify_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        tracker.mlflow.log_artifact(report_path, artifact_path="restoration")

        print("\n[OK] Phase-5 restoration")
        print(f"     PSNR restored:  {rest_result.metrics['psnr_restored']:.2f} dB")
        print(f"     PSNR baseline:  {rest_result.metrics['psnr_baseline_center']:.2f} dB")
        print(f"     PSNR gain:      {rest_result.metrics['psnr_gain_vs_center']:+.2f} dB")
        print(f"     SSIM restored:  {rest_result.metrics['ssim_restored']:.4f}")
        print(f"     Parameters:     {n_params:,}")
        print(f"     Outputs:        {p5.root}")
        print(f"     MLflow:         {tracker.mlflow.run_id}")
    finally:
        tracker.close()

    print("\n=== Phase-5 verification PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
