#!/usr/bin/env python
"""
Phase 5.1 — Train SwinIRUNetHybrid on RLPR (L1 + SSIM + LPIPS vs Pseudo_GT_ROI).

Uses gemma4 CUDA environment. Logs all losses/metrics to MLflow per epoch.

Example::

    D:\\gemma4\\gemma4\\Scripts\\python.exe train_restoration.py
    D:\\gemma4\\gemma4\\Scripts\\python.exe train_restoration.py --max-epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env_paths import ensure_venv_on_path, get_python_executable

ensure_venv_on_path()

import logging

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from datasets.rlpr_restoration_dataset import RLPRRestorationDataset
from engine.restoration_lightning import RestorationLightningModule
from logging_utils.experiment_manager import ExperimentManager
from logging_utils.phase_outputs import PhaseOutputManager
from utils.config_loader import ConfigManager
from utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train restoration model (Phase 5.1).")
    parser.add_argument(
        "--config",
        type=str,
        default="train_restoration.yaml",
        help="Training config file name in configs/",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["center_upscale", "fused_live"],
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ConfigManager(PROJECT_ROOT).loader.load(args.config)
    log_cfg = ConfigManager(PROJECT_ROOT).load("logging", validate=True)

    if args.max_epochs is not None:
        cfg.trainer.max_epochs = args.max_epochs
    if args.input_mode is not None:
        cfg.training.input_mode = args.input_mode

    seed = int(cfg.reproducibility.get("seed", 42))
    set_seed(seed)

    print(f"[INFO] Training with: {get_python_executable()}")
    print(f"[INFO] Input mode: {cfg.training.input_mode}")
    print(f"[INFO] Max epochs: {cfg.trainer.max_epochs}")

    exp_manager = ExperimentManager(PROJECT_ROOT, logging_cfg=log_cfg)
    tracker = exp_manager.create(
        name=str(cfg.experiment.name),
        tags=list(cfg.experiment.get("tags", [])),
        config_snapshot=cfg,
        seed=seed,
    )

    ckpt_dir = tracker.paths.checkpoints
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    phase_dir = PhaseOutputManager(tracker.paths.root).get("restoration_train")
    OmegaConf.save(cfg, tracker.paths.configs / "train_restoration_snapshot.yaml")

    flow_engine = None
    fusion_engine = None
    if cfg.training.input_mode == "fused_live":
        from engine.optical_flow_engine import OpticalFlowEngine
        from engine.temporal_fusion_engine import TemporalFusionEngine
        from logging_utils.phase_outputs import PhaseOutputManager as POM
        from utils.gpu_utils import GPUManager

        gpu = GPUManager.from_config(cfg.hardware)
        cfg.optical_flow.max_frame_pairs = int(cfg.training.get("max_flow_pairs", 6))
        p3 = POM(tracker.paths.root).get("optical_flow")
        p4 = POM(tracker.paths.root).get("temporal_fusion")
        flow_engine = OpticalFlowEngine.from_config(cfg, p3, gpu_manager=gpu)
        fusion_engine = TemporalFusionEngine.from_config(cfg, p4, gpu_manager=gpu)
        print("[INFO] fused_live training: Phase 3+4 enabled (slower).")

    train_ds = RLPRRestorationDataset.from_config(
        cfg, PROJECT_ROOT, split="train", flow_engine=flow_engine, fusion_engine=fusion_engine
    )
    val_ds = RLPRRestorationDataset.from_config(
        cfg, PROJECT_ROOT, split="val", flow_engine=flow_engine, fusion_engine=fusion_engine
    )
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = __import__("torch").utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=True,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
    )
    val_loader = __import__("torch").utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    module = RestorationLightningModule(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=str(log_cfg.mlflow.get("experiment_name", "mf_lpr2")),
        tracking_uri=str(tracker.paths.mlruns.as_uri()),
        run_id=tracker.mlflow.run_id,
        log_model=bool(cfg.get("mlflow", {}).get("log_model", False)),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=str(cfg.checkpointing.get("filename", "restoration_{epoch:02d}")),
            monitor=str(cfg.checkpointing.get("monitor", "val/psnr")),
            mode=str(cfg.checkpointing.get("mode", "max")),
            save_top_k=int(cfg.checkpointing.get("save_top_k", 3)),
            save_last=bool(cfg.checkpointing.get("save_last", True)),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if bool(cfg.early_stopping.get("enabled", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.early_stopping.get("monitor", "val/psnr")),
                mode=str(cfg.early_stopping.get("mode", "max")),
                patience=int(cfg.early_stopping.get("patience", 20)),
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.get("accelerator", "gpu")),
        devices=int(cfg.trainer.get("devices", 1)),
        precision=str(cfg.trainer.get("precision", "16-mixed")),
        accumulate_grad_batches=int(cfg.trainer.get("accumulate_grad_batches", 4)),
        gradient_clip_val=float(cfg.trainer.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(cfg.trainer.get("log_every_n_steps", 1)),
        val_check_interval=float(cfg.trainer.get("val_check_interval", 1.0)),
        deterministic=bool(cfg.trainer.get("deterministic", True)),
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(tracker.paths.root),
    )

    try:
        trainer.fit(module, train_loader, val_loader)
        best = callbacks[0].best_model_path
        print(f"\n[OK] Training complete. Best checkpoint: {best}")
        tracker.mlflow.log_params(
            {
                "best_checkpoint": best or "",
                "train_samples": str(len(train_ds)),
                "val_samples": str(len(val_ds)),
                "input_mode": str(cfg.training.input_mode),
            }
        )
        if best:
            tracker.mlflow.log_artifact(best, artifact_path="checkpoints")
    finally:
        tracker.close()

    print(f"[INFO] MLflow run_id: {tracker.mlflow.run_id}")
    print(f"[INFO] Experiment dir: {tracker.paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
