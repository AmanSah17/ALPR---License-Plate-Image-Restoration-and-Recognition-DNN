"""
Phase 7: Joint Optimization Training Script.

Trains any registered model family using the combined pixel + OCR loss.
Automatically loads the best Phase 4 checkpoint for warm-starting.

Usage:
    # Single model
    python train_joint_optimization.py --model unet_lite --max-epochs 30

    # All models one by one (sequential, no OOM risk)
    python train_joint_optimization.py --multi-run --max-epochs 30

    # Custom loss weights
    python train_joint_optimization.py --model swinir_base --ocr-weight 0.2

    # Smoke test (1 epoch each)
    python train_joint_optimization.py --multi-run --max-epochs 1 --smoke-test

Copyright (c) 2024 Aman Sah (amansah1717@gmail.com)
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import OmegaConf

from engine.joint_lightning import JointOptimizationLightningModule
from datasets.rlpr_restoration_dataset import RLPRRestorationDataset
from utils.config_loader import ConfigManager
from utils.seed_utils import set_seed
from logging_utils.experiment_manager import ExperimentManager
from models.model_registry import initialize_registry, ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_joint_optimization.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── All model families available for joint optimization ─────────────────────
ALL_MODELS = [
    "unet_lite",
    "unet_standard",
    "unet_dense",
    "swinir_small",
    "swinir_base",
    "swinir_large",
    "resnet_small",
    "resnet_medium",
    "spatiotemporal_hybrid_small",
    "spatiotemporal_hybrid_base",
    "hybrid_attention",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 7: Joint Optimization (Restoration + Modular OCR Supervision)"
    )
    parser.add_argument(
        "--model", type=str, default="unet_lite",
        choices=ALL_MODELS,
        help="Model to joint-optimize.",
    )
    parser.add_argument(
        "--multi-run", action="store_true",
        help="Run joint optimization for ALL model families sequentially.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick 1-epoch run for each model to verify no crashes.",
    )
    parser.add_argument("--max-epochs",   type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=1,
                        help="Fixed at 1 for variable-size plates.")
    parser.add_argument("--lr",           type=float, default=5e-5,
                        help="Learning rate for restoration model.")
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--ocr-weight",   type=float, default=0.1,
                        help="Weight for OCR supervision loss.")
    parser.add_argument("--feat-weight",  type=float, default=0.05,
                        help="Weight for OCR perceptual feature loss.")
    parser.add_argument("--train-ocr-backend", type=str, default="parseq",
                        help="Differentiable OCR backend for training loss. Supported: parseq, none.")
    parser.add_argument("--eval-ocr-backend", type=str, default="fast_plate_ocr",
                        help="OCR backend for validation and deployment-style evaluation. Supported: fast_plate_ocr, parseq, none.")
    parser.add_argument("--ocr-loss-type", type=str, default="lcofl", choices=["ce", "lcofl"],
                        help="OCR loss formulation used when the training backend is differentiable.")
    parser.add_argument("--ocr-focal-gamma", type=float, default=2.0,
                        help="Focal gamma used by the LCOFL-style OCR loss.")
    parser.add_argument("--ocr-warmup",   type=int,   default=5,
                        help="Epochs before OCR loss reaches full weight.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--config", type=str,
        default="configs/train_restoration.yaml",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Do not auto-load Phase 4 checkpoint. Train from scratch.",
    )
    parser.add_argument(
        "--experiments-dir", type=str, default="outputs/experiments",
        help="Where Phase 4 checkpoints live.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_ocr_supervisors(
    device: torch.device,
    train_backend: str,
    eval_backend: str,
) -> Tuple['OCRSupervisor', 'OCRValidator']:
    """Load training and validation OCR backends."""
    logger.info("Loading OCR backends (train=%s, eval=%s)...", train_backend, eval_backend)
    from engine.ocr_supervisors import build_ocr_supervisor, build_ocr_validator

    ocr_supervisor = build_ocr_supervisor(train_backend, device)
    val_ocr_validator = build_ocr_validator(eval_backend, device, supervisor=ocr_supervisor)

    logger.info(
        "OCR backends ready. train=%s eval=%s",
        getattr(ocr_supervisor, "backend_name", train_backend),
        getattr(val_ocr_validator, "backend_name", eval_backend),
    )
    return ocr_supervisor, val_ocr_validator


def build_dataloaders(cfg: Any, batch_size: int = 1):
    """Build train/val DataLoaders from config."""
    # Force resize if batch_size > 1 so DataLoader can stack them
    if batch_size > 1:
        if "preprocessing" not in cfg:
            cfg.preprocessing = OmegaConf.create({})
        cfg.preprocessing.resize = [64, 256]
        logger.info("Batch size > 1: Forcing dataset resize to (64, 256) for stacking.")

    # ── Dataset Init ─────────────────────────────────────────────────────
    from datasets.rlpr_restoration_dataset import RLPRRestorationDataset
    from datasets.rlpr_dataset import collate_rlpr_batch
    
    train_ds = RLPRRestorationDataset.from_config(cfg, project_root=".", split="train")
    val_ds = RLPRRestorationDataset.from_config(cfg, project_root=".", split="val", augmentation_enabled=False)
    logger.info(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=collate_rlpr_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=collate_rlpr_batch,
    )
    return train_loader, val_loader


def run_joint_training(
    model_name: str,
    ocr_supervisor: 'OCRSupervisor',
    val_ocr_validator: 'FastPlateOCRValidator',
    cfg: Any,
    args,
    exp_manager: ExperimentManager,
) -> Dict[str, Any]:
    """
    Run joint optimization for a single model family.
    Returns training results dict.
    """
    logger.info("=" * 70)
    logger.info(f"Joint Optimization: {model_name}")
    logger.info("=" * 70)
    
    set_seed(args.seed)
    max_epochs = 1 if args.smoke_test else args.max_epochs
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Build experiment tracker ─────────────────────────────────────────
    log_cfg = ConfigManager(".").load("logging", validate=True)
    tracker = exp_manager.create(
        name=f"joint_p7_{model_name}",
        tags=[model_name, "phase7", "joint_optimization"],
        config_snapshot=cfg,
        seed=args.seed,
    )
    exp_dir = tracker.paths.root

    mlflow_logger = MLFlowLogger(
        experiment_name=str(log_cfg.mlflow.get("experiment_name", "mf_lpr2")),
        tracking_uri=str(tracker.paths.mlruns.as_uri()),
        run_id=tracker.mlflow.run_id,
        log_model=False,
    )

    # ── Build Joint Module ───────────────────────────────────────────────
    pretrained_ckpt = None
    if not args.no_pretrained:
        from engine.joint_lightning import _find_best_checkpoint
        ckpt = _find_best_checkpoint(model_name, args.experiments_dir)
        pretrained_ckpt = str(ckpt) if ckpt else None

    module = JointOptimizationLightningModule(
        model_name=model_name,
        ocr_supervisor=ocr_supervisor,
        val_ocr_validator=val_ocr_validator,
        device_str=args.device,
        pretrained_ckpt=pretrained_ckpt,
        lr=args.lr,
        max_epochs=max_epochs,
        pixel_weight=args.pixel_weight,
        ocr_weight=args.ocr_weight,
        feat_weight=args.feat_weight,
        ocr_loss_type=args.ocr_loss_type,
        ocr_focal_gamma=args.ocr_focal_gamma,
        ocr_warmup_epochs=args.ocr_warmup,
    )

    # ── Override dataset config based on model architecture ──────────────
    import copy
    local_cfg = copy.deepcopy(cfg)
    if model_name.startswith("spatiotemporal_hybrid") or model_name == "hybrid_attention":
        if hasattr(local_cfg, "training"):
            local_cfg.training["input_mode"] = "sequence"
        else:
            local_cfg["input_mode"] = "sequence"
        logger.info(f"Auto-configured input_mode='sequence' for {model_name}")
    else:
        # Explicitly enforce center_upscale for single-frame models
        if hasattr(local_cfg, "training"):
            local_cfg.training["input_mode"] = "center_upscale"
        else:
            local_cfg["input_mode"] = "center_upscale"

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(local_cfg, batch_size=args.batch_size)

    # ── Callbacks ────────────────────────────────────────────────────────
    ckpt_dir = tracker.paths.checkpoints
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save best by CER (minimize) and by PSNR (maximize)
    ckpt_best_cer = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_cer-epoch={epoch:02d}-val_cer={val_cer:.3f}",
        monitor="val_cer",
        mode="min",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )
    ckpt_best_psnr = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_psnr-epoch={epoch:02d}-val_psnr={val_psnr:.3f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=2,
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor="val_cer", mode="min", patience=10, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
        precision="16-mixed" if args.device == "cuda" else 32,  # AMP for OOM prevention
        gradient_clip_val=1.0,
        logger=mlflow_logger,
        callbacks=[ckpt_best_cer, ckpt_best_psnr, early_stop, lr_monitor],
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=False,
    )

    logger.info(f"Starting joint training [{model_name}]...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    results = {
        "model_name": model_name,
        "exp_dir": str(exp_dir),
        "best_cer_ckpt": ckpt_best_cer.best_model_path,
        "best_psnr_ckpt": ckpt_best_psnr.best_model_path,
        "final_epoch": trainer.current_epoch,
    }
    logger.info(f"[SUCCESS] Done [{model_name}] | Best CER ckpt: {ckpt_best_cer.best_model_path}")
    tracker.close()
    return results


def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("Phase 7: Joint Optimization Training")
    logger.info("=" * 70)
    logger.info(f"Device : {args.device}")
    logger.info(f"Seed   : {args.seed}")
    logger.info(f"Epochs : {'1 (SMOKE TEST)' if args.smoke_test else args.max_epochs}")
    logger.info(f"Loss   : pixel={args.pixel_weight}  ocr={args.ocr_weight}  feat={args.feat_weight}")
    logger.info(f"OCR    : train={args.train_ocr_backend}  eval={args.eval_ocr_backend}  loss={args.ocr_loss_type}")

    # ── Load config ───────────────────────────────────────────────────────
    cfg = ConfigManager(".").loader.load(Path(args.config).name)

    # ── Initialize registry ───────────────────────────────────────────────
    initialize_registry()

    # ── Load Modular OCR Supervisors ONCE (shared across all model runs) ────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ocr_supervisor, val_ocr_validator = load_ocr_supervisors(
        device,
        train_backend=args.train_ocr_backend,
        eval_backend=args.eval_ocr_backend,
    )
    if not getattr(ocr_supervisor, "supports_training_loss", True) and args.ocr_weight > 0:
        logger.warning(
            "Training OCR backend '%s' does not support differentiable supervision. "
            "OCR loss will be skipped even though --ocr-weight=%.3f.",
            args.train_ocr_backend,
            args.ocr_weight,
        )

    # ── Determine models to run ────────────────────────────────────────────
    models = ALL_MODELS if args.multi_run else [args.model]
    logger.info(f"Models to train: {models}")

    exp_manager = ExperimentManager(".")
    all_results = []

    for model_name in models:
        # Clear GPU cache between runs to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            mem_before = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory before {model_name}: {mem_before:.2f} GB allocated")

        try:
            results = run_joint_training(
                model_name=model_name,
                ocr_supervisor=ocr_supervisor,
                val_ocr_validator=val_ocr_validator,
                cfg=cfg,
                args=args,
                exp_manager=exp_manager,
            )
            all_results.append(results)
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA OOM for {model_name}. Clearing cache and continuing to next model.")
            torch.cuda.empty_cache()
            gc.collect()
            all_results.append({"model_name": model_name, "error": "CUDA OOM"})
        except Exception as e:
            logger.error(f"Failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"model_name": model_name, "error": str(e)})
        finally:
            # Aggressively free module memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    # ── Final Summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("JOINT OPTIMIZATION SUMMARY")
    logger.info("=" * 70)
    for r in all_results:
        if "error" in r:
            logger.info(f"  [FAIL] {r['model_name']}: FAILED ({r['error']})")
        else:
            logger.info(f"  [OK] {r['model_name']}: epoch={r['final_epoch']}")
            logger.info(f"      Best CER  ckpt: {Path(r['best_cer_ckpt']).name if r['best_cer_ckpt'] else 'N/A'}")
            logger.info(f"      Best PSNR ckpt: {Path(r['best_psnr_ckpt']).name if r['best_psnr_ckpt'] else 'N/A'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
