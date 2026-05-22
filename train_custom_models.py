"""
Main training script for custom restoration model variants.

Supports:
- Multiple model architectures (10 variants across 4 families)
- Comprehensive metrics tracking (8 categories)
- Data augmentation (configurable intensity)
- Multi-run mode for systematic comparison
- MLflow logging and checkpoint management

Usage:
    python train_custom_models.py --model-family swinir --param-scale base
    python train_custom_models.py --multi-run  # Train all models
    python train_custom_models.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.model_registry import ModelRegistry, initialize_registry
from engine.composite_lightning import CompositeRestorationLightningModule
from datasets.rlpr_restoration_dataset import RLPRRestorationDataset
from utils.config_loader import ConfigManager
from utils.seed_utils import set_seed
from logging_utils.experiment_manager import ExperimentManager
from metrics.composite_metrics import ModelComplexity

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the training script."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("train_custom_models.log"),
            logging.StreamHandler(),
        ],
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train custom restoration model variants with comprehensive metrics"
    )

    parser.add_argument(
        "--model-family",
        type=str,
        default="swinir",
        choices=["swinir", "unet", "resnet", "hybrid"],
        help="Model family to train",
    )

    parser.add_argument(
        "--param-scale",
        type=str,
        default="small",
        choices=["small", "base", "large", "medium"],
        help="Parameter scale (size) of the model",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_restoration.yaml",
        help="Path to training config file",
    )

    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Custom experiment name (default: auto-generated)",
    )

    parser.add_argument(
        "--augmentation",
        type=str,
        default="moderate",
        choices=["light", "moderate", "aggressive", "none"],
        help="Data augmentation strength",
    )

    parser.add_argument(
        "--multi-run",
        action="store_true",
        help="Train all models sequentially for comparison",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (note: fixed at 1 for variable-size plates)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda or cpu)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--profile-runtime",
        action="store_true",
        help="Profile runtime metrics (FPS, latency, VRAM) after training",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def build_model_name(family: str, scale: str) -> str:
    """Build model name from family and scale."""
    if family == "hybrid":
        return "hybrid_attention"
    if family == "unet":
        if scale == "small" or scale == "lite":
            return "unet_lite"
        elif scale == "base" or scale == "standard":
            return "unet_standard"
        elif scale == "large" or scale == "dense":
            return "unet_dense"
    return f"{family}_{scale}"


def train_model_variant(
    model_name: str,
    cfg: Any,
    augmentation_strength: str,
    max_epochs: int,
    device: str,
    seed: int,
    profile_runtime: bool = False,
    exp_manager: Optional[ExperimentManager] = None,
) -> Dict[str, Any]:
    """
    Train a single model variant.

    Args:
        model_name: Name of model to train (must be registered)
        cfg: Training configuration
        augmentation_strength: Augmentation intensity
        max_epochs: Maximum number of epochs
        device: Device to train on
        seed: Random seed
        profile_runtime: Whether to profile runtime
        exp_manager: Experiment manager for organizing outputs

    Returns:
        Dictionary with training results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training model variant: {model_name}")
    logger.info(f"{'='*80}")

    # Set seed
    set_seed(seed)

    # Load logging config
    log_cfg = ConfigManager(".").load("logging", validate=True)
    if exp_manager is None:
        exp_manager = ExperimentManager(".", logging_cfg=log_cfg)

    # Create experiment tracker
    tracker = exp_manager.create(
        name=f"train_{model_name}",
        tags=[model_name, "custom_training"],
        config_snapshot=cfg,
        seed=seed,
    )
    exp_dir = tracker.paths.root

    # Setup MLflow logger linked to the tracker's run
    mlflow_logger = MLFlowLogger(
        experiment_name=str(log_cfg.mlflow.get("experiment_name", "mf_lpr2")),
        tracking_uri=str(tracker.paths.mlruns.as_uri()),
        run_id=tracker.mlflow.run_id,
        log_model=False,
    )

    # Load training configuration and update max_epochs
    from omegaconf import DictConfig, OmegaConf
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
        if "trainer" not in cfg:
            cfg.trainer = {}
        cfg.trainer.max_epochs = max_epochs
    elif isinstance(cfg, dict):
        if "trainer" not in cfg:
            cfg["trainer"] = {}
        cfg["trainer"]["max_epochs"] = max_epochs
    else:
        try:
            if hasattr(cfg, "trainer"):
                if isinstance(cfg.trainer, dict):
                    cfg.trainer["max_epochs"] = max_epochs
                else:
                    cfg.trainer.max_epochs = max_epochs
            else:
                cfg["trainer"] = {"max_epochs": max_epochs}
        except Exception as e:
            logger.warning(f"Could not set max_epochs on config object: {e}")

    try:
        # Build model from registry
        logger.info(f"Building model: {model_name}")
        model = ModelRegistry.build(model_name)
        
        # Log model complexity
        num_params = ModelComplexity.count_parameters(model)
        flops = ModelComplexity.estimate_flops(model, (1, 3, 256, 256))
        complexity_str = ModelComplexity.format_complexity(num_params, flops)
        logger.info(f"Model complexity: {complexity_str}")
        
        try:
            tracker.mlflow.log_params({
                "model_family": model_name.split("_")[0],
                "model_num_params": num_params,
                "model_flops": flops or 0,
            })
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

        # Create Lightning module
        lightning_module = CompositeRestorationLightningModule(
            cfg,
            enable_ocr_metrics=True,
            enable_runtime_profiling=profile_runtime,
            model_name=model_name,
            model=model,
        )

        # Load dataset with augmentation
        logger.info("Loading dataset with augmentation...")
        augment_enabled = augmentation_strength != "none"
        train_dataset = RLPRRestorationDataset.from_config(
            cfg,
            project_root=".",
            split="train",
            augmentation_enabled=augment_enabled,
            augmentation_strength=augmentation_strength if augment_enabled else "moderate",
        )
        val_dataset = RLPRRestorationDataset.from_config(
            cfg,
            project_root=".",
            split="val",
            augmentation_enabled=False,  # No augmentation on val
        )

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # Setup callbacks
        checkpoint_dir = tracker.paths.checkpoints
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:02d}-{val_psnr:.3f}",
            monitor="val_psnr",
            mode="max",
            save_top_k=3,
            save_last=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_psnr",
            mode="max",
            patience=20,
            verbose=True,
        )

        # Create Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if device == "cuda" else "cpu",
            devices=1,
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True,
        )

        # Train
        logger.info("Starting training...")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Profile runtime if requested
        runtime_metrics = {}
        if profile_runtime:
            logger.info("Profiling runtime metrics...")
            try:
                runtime_metrics = lightning_module.profile_runtime(
                    input_shape=(1, 3, 256, 256),
                    num_runs=10,
                )
                for key, val in runtime_metrics.items():
                    try:
                        tracker.mlflow.log_metric(f"runtime/{key}", val)
                    except Exception as e:
                        logger.warning(f"Failed to log metric to MLflow: {e}")
            except Exception as e:
                logger.warning(f"Failed to profile runtime: {e}")

        # Export metrics
        logger.info("Exporting training metrics...")
        try:
            metrics_files = lightning_module.export_metrics(str(exp_dir / "metrics"))
            logger.info(f"Exported metrics: {metrics_files}")
        except Exception as e:
            logger.warning(f"Failed to export metrics: {e}")

        # Prepare results
        results = {
            "model_name": model_name,
            "exp_dir": str(exp_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "num_params": num_params,
            "flops": flops,
            "final_epoch": trainer.current_epoch,
            "best_checkpoint": checkpoint_callback.best_model_path,
            "runtime_metrics": runtime_metrics,
        }

        logger.info(f"\n✓ Training complete for {model_name}")
        logger.info(f"  Best checkpoint: {results['best_checkpoint']}")
        logger.info(f"  Experiment dir: {exp_dir}")

        return results
    finally:
        tracker.close()


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.log_level)

    logger.info("="*80)
    logger.info("Custom Restoration Model Training")
    logger.info("="*80)
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")

    # Initialize model registry
    logger.info("Initializing model registry...")
    initialize_registry()
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config_file = Path(args.config).name
    cfg = ConfigManager(".").loader.load(config_file)

    # Create experiment manager
    exp_manager = ExperimentManager(".")

    # Determine models to train
    if args.multi_run:
        logger.info("Multi-run mode: Training all registered models")
        models_to_train = list(ModelRegistry.list_models().keys())
    else:
        # Build model name from family and scale
        model_name = build_model_name(args.model_family, args.param_scale)
        if model_name not in ModelRegistry.list_models():
            logger.error(f"Unknown model: {model_name}")
            logger.info(f"Available models:\n{ModelRegistry.print_summary()}")
            sys.exit(1)
        models_to_train = [model_name]

    # Print registry summary
    logger.info(f"\n{ModelRegistry.print_summary()}\n")

    # Train all models
    results_list = []
    for model_name in models_to_train:
        try:
            results = train_model_variant(
                model_name=model_name,
                cfg=cfg,
                augmentation_strength=args.augmentation if args.augmentation != "none" else "moderate",
                max_epochs=args.max_epochs,
                device=args.device,
                seed=args.seed,
                profile_runtime=args.profile_runtime,
                exp_manager=exp_manager,
            )
            results_list.append(results)
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    for results in results_list:
        logger.info(f"\nModel: {results['model_name']}")
        logger.info(f"  Params: {results['num_params'] / 1e3:.1f}K")
        if results['flops']:
            logger.info(f"  FLOPs: {results['flops'] / 1e9:.2f}G")
        logger.info(f"  Best checkpoint: {Path(results['best_checkpoint']).name}")
        if results['runtime_metrics']:
            logger.info(f"  Runtime: FPS={results['runtime_metrics'].get('fps', 0):.1f}, "
                       f"Latency={results['runtime_metrics'].get('latency_ms', 0):.1f}ms")

    logger.info("\n" + "="*80)
    logger.info("Training completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
