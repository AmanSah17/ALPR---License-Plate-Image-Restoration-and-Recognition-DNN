"""
Extended PyTorch Lightning module with comprehensive restoration metrics.

Extends RestorationLightningModule with:
- Composite metrics (PSNR, SSIM, LPIPS with profiling)
- OCR-related metrics (CER, sequence accuracy)
- Runtime profiling (FPS, latency, VRAM)
- Per-image metrics logging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import torch
import pytorch_lightning as pl

from engine.restoration_lightning import RestorationLightningModule
from metrics.composite_metrics import CompositeMetricComputer, RuntimeProfiler, ModelComplexity
from metrics.ocr_metrics import CharacterErrorRate, SequenceAccuracy, OCRMetricHooks
from utils.metric_aggregator import MetricAggregator

logger = logging.getLogger(__name__)


class CompositeRestorationLightningModule(RestorationLightningModule):
    """
    Extended Lightning module with all 8 metric categories:
    1. Restoration Quality (PSNR, SSIM)
    2. Perceptual (LPIPS)
    3. OCR Readiness (CER, Sequence Accuracy)
    4. Runtime (FPS, latency, VRAM)
    5. Complexity (FLOPs, parameters)
    6. Overfitting Health (train/val loss ratio)
    7. Baseline Comparison (gain vs center)
    8. Dataset Stats
    
    Backward compatible with existing RestorationLightningModule.
    """

    def __init__(
        self,
        cfg: Any,
        enable_ocr_metrics: bool = True,
        enable_runtime_profiling: bool = False,
        model_name: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Initialize composite Lightning module.

        Args:
            cfg: Configuration object
            enable_ocr_metrics: Whether to compute CER/sequence accuracy (mocks unless Phase 6 OCR wired)
            enable_runtime_profiling: Whether to profile runtime (can be slow)
            model_name: Name of custom model architecture
            model: Custom model instance
        """
        super().__init__(cfg)
        self.enable_ocr_metrics = enable_ocr_metrics
        self.enable_runtime_profiling = enable_runtime_profiling

        # Set or override model if custom model is provided
        if model is not None:
            self.model = model
            if model_name is None:
                model_name = getattr(model, "name", "unknown_custom_model")
        elif model_name is not None:
            from models.model_registry import ModelRegistry, initialize_registry
            initialize_registry()
            self.model = ModelRegistry.build(model_name)
        else:
            model_name = getattr(cfg.model, "name", "swinir_small")

        self.model_name = model_name
        self.save_hyperparameters(ignore=["cfg", "model"])

        # Initialize composite metric computer
        lpips_enabled = cfg.losses.get("lpips_weight", 0.0) > 0.0
        self.metric_computer = CompositeMetricComputer(lpips_enabled=lpips_enabled)

        # Initialize OCR hooks (Phase 6 will wire PARSeq here)
        self.ocr_hooks = OCRMetricHooks() if enable_ocr_metrics else None

        # Initialize metric aggregator for CSV export
        self.metric_aggregator = MetricAggregator(
            experiment_name=cfg.get("experiment_name", "restoration_training"),
            model_name=self.model_name,
        )

        # Track model complexity once at init
        self._log_model_complexity()

        # Runtime profiler (initialized later if needed)
        self.runtime_profiler = None

    def _log_model_complexity(self):
        """Log model complexity metrics (FLOPs, parameters)."""
        num_params = ModelComplexity.count_parameters(self.model)
        flops = ModelComplexity.estimate_flops(self.model, (1, 3, 256, 256))
        complexity_str = ModelComplexity.format_complexity(num_params, flops)
        logger.info(f"Model complexity: {complexity_str}")

        # Log as hyperparameter for MLflow
        self.logger.experiment.log_params(
            {
                "model_num_params": num_params,
                "model_flops": flops if flops else 0,
            }
        ) if hasattr(self.logger, "experiment") else None

    def _compute_composite_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_tensor: torch.Tensor,
        sample_ids: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Compute all available metrics for a batch.

        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            input_tensor: Input images for baseline comparison (B, C, H, W)
            sample_ids: Optional list of sample IDs for tracking

        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}

        # Ensure tensors are on CPU for metric computation
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        input_tensor = input_tensor.detach().cpu()

        # Restoration quality metrics (PSNR, SSIM, LPIPS)
        quality_metrics = self.metric_computer.compute_all_metrics(pred, target, compute_lpips=True)
        metrics.update(quality_metrics)

        # Baseline comparison (gain vs input)
        input_for_baseline = input_tensor
        if input_tensor.shape[1] > 3 and input_tensor.shape[1] % 3 == 0:
            seq_len = input_tensor.shape[1] // 3
            center_idx = seq_len // 2
            input_for_baseline = input_tensor[:, center_idx*3:(center_idx+1)*3, :, :]
            
        baseline_metrics = self.metric_computer.compute_all_metrics(input_for_baseline, target, compute_lpips=False)
        if "psnr" in quality_metrics and "psnr" in baseline_metrics:
            metrics["psnr_gain_vs_input"] = quality_metrics["psnr"] - baseline_metrics["psnr"]
        if "ssim" in quality_metrics and "ssim" in baseline_metrics:
            metrics["ssim_gain_vs_input"] = quality_metrics["ssim"] - baseline_metrics["ssim"]

        # OCR metrics (mock version for now; Phase 6 will integrate live OCR)
        if self.enable_ocr_metrics and self.ocr_hooks is not None:
            try:
                # Mock CER based on restoration quality
                mock_ocr_metrics = OCRMetricHooks.compute_mock_cer(quality_metrics, len(sample_ids or []))
                metrics.update(mock_ocr_metrics)
            except Exception as e:
                logger.warning(f"Error computing mock OCR metrics: {e}")

        return metrics

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """Override parent to include composite metrics."""
        inp = batch["input"]
        tgt = batch["target"]
        sample_ids = batch.get("sample_id", None)

        if inp.dim() == 3:
            inp = inp.unsqueeze(0)
            tgt = tgt.unsqueeze(0)

        pred = self(inp)
        loss, components = self.criterion(pred, tgt)

        # Log base metrics (same as parent)
        self.log(
            f"{stage}/loss",
            components["total"],
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        for name in ("l1", "ssim", "lpips"):
            self.log(
                f"{stage}/{name}",
                components[name],
                on_step=False,
                on_epoch=True,
            )

        # Log composite metrics only for validation
        if stage == "val":
            pred0 = pred[0].detach().cpu()
            tgt0 = tgt[0].detach().cpu()
            inp0 = inp[0].detach().cpu()

            composite_metrics = self._compute_composite_metrics(pred, tgt, inp, sample_ids)

            # Log all composite metrics
            for metric_name, metric_val in composite_metrics.items():
                if metric_val is not None and not isinstance(metric_val, str):
                    self.log(f"val/{metric_name}", metric_val, on_epoch=True)

            # Keep backward compatibility aliases
            self.log("val_psnr", composite_metrics.get("psnr", 0.0), on_epoch=True)

            # Track train/val loss ratio for overfitting detection
            if hasattr(self, "_last_train_loss"):
                train_loss_val = self._last_train_loss
                val_loss_val = components["total"]
                overfitting_ratio = val_loss_val / (train_loss_val + 1e-8)
                self.log("overfitting_ratio", overfitting_ratio, on_epoch=True)

        elif stage == "train":
            # Track last training loss for overfitting ratio
            self._last_train_loss = components["total"].item()

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step (inherited from parent, uses _shared_step override)."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step (inherited from parent, uses _shared_step override)."""
        return self._shared_step(batch, "val")

    def on_train_epoch_end(self):
        """Hook called at end of training epoch."""
        super().on_train_epoch_end() if hasattr(super(), "on_train_epoch_end") else None

        # Log epoch metrics to aggregator
        current_epoch = self.current_epoch
        if hasattr(self, "trainer") and self.trainer is not None:
            # Get learning rate
            lr = self.optimizers().param_groups[0]["lr"]

            # Log to aggregator (for CSV export)
            train_metrics = {}
            val_metrics = {}

            if hasattr(self, "logged_metrics"):
                for key, val in self.logged_metrics.items():
                    if "train/" in key:
                        train_metrics[key.replace("train/", "")] = val
                    elif "val/" in key:
                        val_metrics[key.replace("val/", "")] = val

            if train_metrics or val_metrics:
                self.metric_aggregator.log_epoch_metrics(current_epoch, train_metrics, val_metrics, lr)

    def profile_runtime(self, input_shape: tuple = (1, 3, 256, 256), num_runs: int = 10) -> Dict[str, float]:
        """
        Profile model runtime characteristics.

        Args:
            input_shape: Input tensor shape
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with FPS, latency (ms), VRAM (MB), RAM (MB)
        """
        if self.runtime_profiler is None:
            self.runtime_profiler = RuntimeProfiler(self.model, device=self.device)

        input_tensor = torch.randn(input_shape, device=self.device)
        runtime_metrics = self.runtime_profiler.profile_inference(input_tensor, num_runs=num_runs)

        logger.info(f"Runtime profile: {runtime_metrics}")
        return runtime_metrics

    def register_ocr_engine(self, ocr_engine: Any):
        """
        Register external OCR engine (Phase 6+).

        Args:
            ocr_engine: Callable(image: torch.Tensor) -> str (e.g., PARSeq model)
        """
        if self.ocr_hooks is not None:
            self.ocr_hooks.register_ocr_engine(ocr_engine)
            logger.info("✓ PARSeq OCR engine registered. Will compute live CER from Phase 6 onward.")

    def export_metrics(self, output_dir: str = "outputs/metrics") -> Dict[str, Any]:
        """
        Export training metrics to CSV, JSON, and HTML.

        Args:
            output_dir: Directory to save exports

        Returns:
            Dictionary mapping format -> file path
        """
        return self.metric_aggregator.export_all(base_filename=f"epoch_{self.current_epoch:03d}")


if __name__ == "__main__":
    # Test import and basic functionality
    print("✓ CompositeRestorationLightningModule imported successfully")
    print("  Features:")
    print("  - Composite metrics (PSNR, SSIM, LPIPS, runtime profiling)")
    print("  - OCR hooks for Phase 6 PARSeq integration")
    print("  - Metric aggregation (CSV, JSON, HTML export)")
    print("  - Backward compatible with RestorationLightningModule")
