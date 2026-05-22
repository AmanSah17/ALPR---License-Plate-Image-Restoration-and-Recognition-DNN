"""
Restoration engine: fused latent -> SwinIR-UNet -> metrics vs pseudo-GT ROI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from logging_utils.phase_outputs import PhaseOutputPaths
from metrics.psnr import compute_psnr
from metrics.ssim import compute_ssim
from models.restoration.swinir_unet import SwinIRUNetHybrid
from utils.gpu_utils import GPUManager
from utils.image_utils import save_image_rgb
from utils.profiler import count_parameters, estimate_flops
from utils.timing import BenchmarkTimer

logger = logging.getLogger(__name__)


@dataclass
class RestorationResult:
    """Restoration outputs and quality metrics."""

    sample_id: str
    restored: torch.Tensor  # (3, H, W)
    baseline_center: torch.Tensor  # center frame upsampled
    target: torch.Tensor  # pseudo GT ROI
    metrics: Dict[str, float] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)


class RestorationEngine:
    """
    Apply SwinIR-UNet restoration on Phase-4 fused latent.

    Compares restored output against:
        - Pseudo_GT_ROI (primary supervision target)
        - Upsampled center frame baseline (no restoration ablation)
    """

    def __init__(
        self,
        model: SwinIRUNetHybrid,
        output_paths: PhaseOutputPaths,
        gpu_manager: Optional[GPUManager] = None,
        mixed_precision: bool = True,
        upscale_to_gt: bool = True,
        save_restored_png: bool = True,
        save_error_maps: bool = True,
    ) -> None:
        self.model = model
        self.outputs = output_paths
        self.gpu = gpu_manager or GPUManager()
        self.mixed_precision = mixed_precision
        self.upscale_to_gt = upscale_to_gt
        self.save_restored_png = save_restored_png
        self.save_error_maps = save_error_maps
        self.model = self.gpu.move_module(self.model)
        self.model.eval()

    def _prepare_input(
        self,
        fused: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Upscale fused CHW tensor to target (H, W) if configured."""
        if not self.upscale_to_gt:
            return fused
        h, w = target_hw
        return F.interpolate(
            fused.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    @torch.inference_mode()
    def restore(self, fused: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Run restoration model.

        Args:
            fused: ``(3, H, W)`` fused latent [0,1].
            target: ``(3, H_gt, W_gt)`` GT for sizing.

        Returns:
            Restored ``(3, H_gt, W_gt)``.
        """
        inp = self._prepare_input(fused, (target.shape[1], target.shape[2]))
        batch = self.gpu.to_device(inp.unsqueeze(0))
        with self.gpu.autocast_context(enabled=self.mixed_precision):
            out = self.model(batch)
        return out.squeeze(0).float().cpu()

    def run(
        self,
        fused: torch.Tensor,
        center_frame: torch.Tensor,
        pseudo_gt_roi: torch.Tensor,
        sample_id: str,
    ) -> RestorationResult:
        """
        Full restoration pass with metrics and artifact export.

        Args:
            fused: Phase-4 fused image CHW [0,1] at center resolution.
            center_frame: Center frame CHW [0,1].
            pseudo_gt_roi: Ground-truth ROI uint8 or float CHW.
            sample_id: Sample name for files.

        Returns:
            ``RestorationResult``.
        """
        if pseudo_gt_roi.dtype == torch.uint8 or pseudo_gt_roi.max() > 1.0:
            target = pseudo_gt_roi.float() / 255.0
        else:
            target = pseudo_gt_roi.float()
        if target.dim() == 3 and target.shape[0] != 3:
            target = target.permute(2, 0, 1)

        timer = BenchmarkTimer("restore", warmup=0)
        with timer.measure():
            restored = self.restore(fused, target)

        baseline = self._prepare_input(center_frame, (target.shape[1], target.shape[2]))

        metrics = {
            "psnr_restored": compute_psnr(restored, target),
            "ssim_restored": compute_ssim(restored, target),
            "psnr_baseline_center": compute_psnr(baseline, target),
            "ssim_baseline_center": compute_ssim(baseline, target),
            "psnr_gain_vs_center": 0.0,
            "ssim_gain_vs_center": 0.0,
            "inference_ms": timer.stats().mean_ms,
            "num_parameters": float(count_parameters(self.model)),
        }
        metrics["psnr_gain_vs_center"] = metrics["psnr_restored"] - metrics["psnr_baseline_center"]
        metrics["ssim_gain_vs_center"] = metrics["ssim_restored"] - metrics["ssim_baseline_center"]

        paths = self._save_artifacts(sample_id, restored, baseline, target, metrics)

        logger.info(
            "[%s] PSNR restored=%.2f dB (baseline=%.2f, gain=%.2f)",
            sample_id,
            metrics["psnr_restored"],
            metrics["psnr_baseline_center"],
            metrics["psnr_gain_vs_center"],
        )

        return RestorationResult(
            sample_id=sample_id,
            restored=restored,
            baseline_center=baseline,
            target=target,
            metrics=metrics,
            artifact_paths=paths,
        )

    def _save_artifacts(
        self,
        sample_id: str,
        restored: torch.Tensor,
        baseline: torch.Tensor,
        target: torch.Tensor,
        metrics: Dict[str, float],
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        restored_dir = self.outputs.root / "restored"
        comparison_dir = self.outputs.root / "comparisons"
        error_dir = self.outputs.error_maps if hasattr(self.outputs, "error_maps") else self.outputs.root / "error_maps"
        restored_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        def _save_chw(tensor: torch.Tensor, path: Path) -> None:
            arr = (tensor.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            save_image_rgb(arr, path)

        if self.save_restored_png:
            p = restored_dir / f"{sample_id}_restored.png"
            _save_chw(restored, p)
            paths["restored"] = str(p)

        comp_path = comparison_dir / f"{sample_id}_gt_baseline_restored.png"
        self._save_triptych(target, baseline, restored, comp_path)
        paths["comparison"] = str(comp_path)

        if self.save_error_maps:
            err_r = torch.mean(torch.abs(restored - target), dim=0)
            err_b = torch.mean(torch.abs(baseline - target), dim=0)
            er = (err_r / (err_r.max() + 1e-8) * 255).numpy().astype(np.uint8)
            eb = (err_b / (err_b.max() + 1e-8) * 255).numpy().astype(np.uint8)
            ep = error_dir / f"{sample_id}_error_restored.png"
            ebp = error_dir / f"{sample_id}_error_baseline.png"
            save_image_rgb(er, ep)
            save_image_rgb(eb, ebp)
            paths["error_restored"] = str(ep)
            paths["error_baseline"] = str(ebp)

        metrics_path = self.outputs.metrics / f"{sample_id}_restoration_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        paths["metrics"] = str(metrics_path)
        return paths

    @staticmethod
    def _save_triptych(
        target: torch.Tensor,
        baseline: torch.Tensor,
        restored: torch.Tensor,
        path: Path,
    ) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(
            axes,
            [target, baseline, restored],
            ["Pseudo GT ROI", "Center (upsampled)", "Restored"],
        ):
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def load_checkpoint(self, path: Path) -> None:
        """Load trained weights from Lightning .ckpt or plain state dict."""
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
            # Lightning prefixes keys with "model."
            state = {
                (k[6:] if k.startswith("model.") else k): v for k, v in state.items()
            }
        else:
            state = raw
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded restoration checkpoint: %s (missing=%d, unexpected=%d)",
            path,
            len(missing),
            len(unexpected),
        )

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        output_paths: PhaseOutputPaths,
        gpu_manager: Optional[GPUManager] = None,
    ) -> "RestorationEngine":
        gpu = gpu_manager or GPUManager.from_config(cfg.hardware)
        
        ckpt = cfg.restoration.get("checkpoint_path")
        model = None
        model_name = None
        
        if ckpt:
            ckpt_path = Path(ckpt)
            if ckpt_path.exists():
                try:
                    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    hparams = checkpoint.get("hyper_parameters", {})
                    if hparams and "model_name" in hparams:
                        model_name = hparams["model_name"]
                        logger.info(f"Detected model '{model_name}' in checkpoint hyperparameters metadata.")
                except Exception as e:
                    logger.warning(f"Failed to extract model name from checkpoint: {e}")
        
        if model_name is None:
            rest_section = cfg.restoration if hasattr(cfg, "restoration") else cfg
            model_name = rest_section.get("model_name", None)
            
        if model_name is not None:
            try:
                from models.model_registry import ModelRegistry, initialize_registry
                initialize_registry()
                model = ModelRegistry.build(model_name)
            except Exception as e:
                logger.warning(f"Failed to build registered model '{model_name}': {e}. Falling back to default.")
                
        if model is None:
            model = SwinIRUNetHybrid.from_config(cfg)
            
        engine = cls(
            model=model,
            output_paths=output_paths,
            gpu_manager=gpu,
            mixed_precision=bool(cfg.inference.get("mixed_precision", True)),
            upscale_to_gt=bool(cfg.inference.get("upscale_to_gt", True)),
            save_restored_png=bool(cfg.output.get("save_restored_png", True)),
            save_error_maps=bool(cfg.output.get("save_error_maps", True)),
        )
        if ckpt:
            engine.load_checkpoint(Path(ckpt))
        return engine
