"""
Phase 7: Joint Optimization Lightning Module.

Orchestrates end-to-end training where gradients from the PARSeq OCR engine
flow backwards into the restoration / refinement networks.

Architecture:
    Input frames
         ↓
    Restoration Model (trainable - UNet / SwinIR / Spatiotemporal)
         ↓
    OCR Refinement Hook (trainable - edge-emphasis)
         ↓
    PARSeq (FROZEN - OCR oracle)
         ↓
    Joint Loss = λ_pixel * L1+SSIM  +  λ_ocr * CrossEntropy  +  λ_feat * PercOCR

Copyright (c) 2024 Aman Sah (amansah1717@gmail.com)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from models.model_registry import ModelRegistry, initialize_registry
from models.restoration.refinement import OCRRefinementHook
from losses.joint_loss import JointLoss, JointLossWeights
from metrics.psnr import compute_psnr
from metrics.ssim import compute_ssim
import Levenshtein

logger = logging.getLogger(__name__)


def _find_best_checkpoint(model_name: str, experiments_dir: str = "outputs/experiments") -> Optional[Path]:
    """
    Scan experiments directory and return the absolute best checkpoint for model_name.
    Matches directory names containing model_name and picks highest PSNR checkpoint.
    """
    base = Path(experiments_dir)
    if not base.exists():
        return None
    # Find all experiment dirs matching model_name
    candidates = sorted(base.glob(f"*{model_name}*"))
    if not candidates:
        return None
    # Take latest experiment (highest timestamp prefix)
    latest_dir = candidates[-1]
    ckpt_dir = latest_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    best_files = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not best_files:
        last = ckpt_dir / "last.ckpt"
        return last if last.exists() else None
    # The last in sorted order has the highest PSNR (embedded in filename)
    return best_files[-1]


class JointOptimizationLightningModule(pl.LightningModule):
    """
    Phase 7: Joint restoration + OCR optimization.

    Supports all model families:
        - unet_lite, unet_standard, unet_dense
        - swinir_small, swinir_base, swinir_large
        - resnet_small, resnet_medium
        - spatiotemporal_hybrid_small, spatiotemporal_hybrid_base
        - hybrid_attention
    """

    def __init__(
        self,
        model_name: str,
        ocr_supervisor: 'OCRSupervisor',
        val_ocr_validator: 'FastPlateOCRValidator',
        device_str: str = "cuda",
        pretrained_ckpt: Optional[str] = None,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        warmup_epochs: int = 3,
        pixel_weight: float = 1.0,
        ocr_weight: float = 0.1,
        feat_weight: float = 0.05,
        ocr_warmup_epochs: int = 5,
        gradient_clip_val: float = 1.0,
        use_gradient_checkpointing: bool = False,
        refinement_enabled: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ocr_supervisor", "val_ocr_validator"])
        
        self.model_name = model_name
        self.ocr_supervisor = ocr_supervisor
        self.val_ocr_validator = val_ocr_validator
        
        # 1. Initialize restoration model from registry
        initialize_registry()
        self.restoration_model = ModelRegistry.build(model_name)
        if use_gradient_checkpointing and hasattr(self.restoration_model, "gradient_checkpointing"):
            self.restoration_model.gradient_checkpointing = True

        # Load pretrained weights if provided (Phase 4 best checkpoint)
        if pretrained_ckpt and Path(pretrained_ckpt).exists():
            logger.info(f"Loading pretrained weights from {pretrained_ckpt}")
            raw = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            state = raw.get("state_dict", raw)
            state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
            missing, unexpected = self.restoration_model.load_state_dict(state, strict=False)
            logger.info(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Gradient checkpointing to save VRAM
        if use_gradient_checkpointing:
            try:
                self.restoration_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            except AttributeError:
                logger.debug("Model does not support gradient_checkpointing_enable(), skipping")

        # ── 2. OCR Refinement Hook (trainable) ──────────────────────────────
        self.refinement = OCRRefinementHook(channels=3, enabled=refinement_enabled)

        # ── 3. Joint Loss ───────────────────────────────────────────────────
        dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.criterion = JointLoss(
            ocr_supervisor=ocr_supervisor,
            device=dev,
            weights=JointLossWeights(
                pixel_weight=pixel_weight,
                ocr_weight=ocr_weight,
                feat_weight=feat_weight,
                ocr_warmup_epochs=ocr_warmup_epochs
            )
        )

        # ── Tracking ────────────────────────────────────────────────────────
        self._val_preds: List[str] = []
        self._val_gts: List[str] = []

        logger.info(f"JointOptimizationLightningModule initialized: {model_name}")
        num_params = sum(p.numel() for p in self.restoration_model.parameters())
        logger.info(f"  Restoration params: {num_params / 1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        restored = self.restoration_model(x)
        refined  = self.refinement(restored, restored)
        return torch.clamp(refined, 0.0, 1.0)

    def on_train_epoch_start(self):
        """Update loss annealing schedule each epoch."""
        self.criterion.set_epoch(self.current_epoch)
        self.parseq.eval()  # Keep PARSeq frozen throughout

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        inp  = batch["input"]
        tgt  = batch["target"]
        texts = batch.get("plate_text_compact", [""] * inp.shape[0])
        if isinstance(texts, str):
            texts = [texts]

        if inp.dim() == 3:
            inp = inp.unsqueeze(0)
            tgt = tgt.unsqueeze(0)

        # ── Resize input to GT size (variable-shape ROI plates) ──────────
        tgt_h, tgt_w = tgt.shape[-2], tgt.shape[-1]
        if inp.shape[-2] != tgt_h or inp.shape[-1] != tgt_w:
            inp = F.interpolate(inp, (tgt_h, tgt_w), mode="bilinear", align_corners=False)

        pred = self.forward(inp)

        # Resize pred if still mismatched (e.g. model adds padding)
        if pred.shape[-2:] != tgt.shape[-2:]:
            pred = F.interpolate(pred, tgt.shape[-2:], mode="bilinear", align_corners=False)

        total, components = self.criterion(pred, tgt, texts)

        self.log("train/loss",     components["total"],   prog_bar=True, on_step=True,  on_epoch=True)
        self.log("train/l1",       components["l1"],      on_step=False, on_epoch=True)
        self.log("train/ssim",     components["ssim"],    on_step=False, on_epoch=True)
        self.log("train/ocr_loss", components.get("ocr", torch.tensor(0.0)),  on_step=False, on_epoch=True)
        self.log("train/feat_loss",components.get("feat", torch.tensor(0.0)), on_step=False, on_epoch=True)
        self.log("train/ocr_scale",components.get("ocr_scale", torch.tensor(0.0)), on_step=False, on_epoch=True)
        return total

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        inp   = batch["input"]
        tgt   = batch["target"]
        texts = batch.get("plate_text_compact", [""] * inp.shape[0])
        if isinstance(texts, str):
            texts = [texts]

        if inp.dim() == 3:
            inp = inp.unsqueeze(0)
            tgt = tgt.unsqueeze(0)

        tgt_h, tgt_w = tgt.shape[-2], tgt.shape[-1]
        if inp.shape[-2] != tgt_h or inp.shape[-1] != tgt_w:
            inp = F.interpolate(inp, (tgt_h, tgt_w), mode="bilinear", align_corners=False)

        pred = self.forward(inp)
        if pred.shape[-2:] != tgt.shape[-2:]:
            pred = F.interpolate(pred, tgt.shape[-2:], mode="bilinear", align_corners=False)

        # ── PSNR / SSIM ──────────────────────────────────────────────────
        p0 = pred[0].cpu()
        t0 = tgt[0].cpu()
        psnr = compute_psnr(p0, t0)
        ssim = compute_ssim(p0, t0)

        # ── Modular OCR decode ────────────────────────────────────────────
        # FastPlateOCR Validator is preferred for Sequence Accuracy validation
        if self.val_ocr_validator is not None and self.val_ocr_validator.recognizer is not None:
            labels = self.val_ocr_validator.decode_images(pred)
        else:
            # Fallback to the training OCR supervisor
            logits = self.ocr_supervisor(pred)
            labels = self.ocr_supervisor.decode_logits(logits)
            
        self._val_preds.extend(labels)
        self._val_gts.extend(texts)

        self.log("val/psnr", psnr, prog_bar=True, on_epoch=True)
        self.log("val_psnr", psnr, on_epoch=True)   # monitored by ModelCheckpoint
        self.log("val/ssim", ssim, on_epoch=True)

    def on_validation_epoch_end(self):
        """Compute aggregate CER and sequence accuracy at epoch end."""
        if not self._val_preds:
            return
        
        cer_sum = 0.0
        exact_match = 0
        for pred, gt in zip(self._val_preds, self._val_gts):
            pred_c = pred.lower().replace(" ", "")
            gt_c   = gt.lower().replace(" ", "")
            if len(gt_c) == 0:
                continue
            cer_sum += Levenshtein.distance(pred_c, gt_c) / len(gt_c)
            if pred_c == gt_c:
                exact_match += 1

        n = len(self._val_preds)
        mean_cer = cer_sum / max(n, 1)
        seq_acc  = exact_match / max(n, 1)

        self.log("val/cer",     mean_cer, prog_bar=True, on_epoch=True)
        self.log("val_cer",     mean_cer, on_epoch=True)  # monitored by ModelCheckpoint (minimize)
        self.log("val/seq_acc", seq_acc,  prog_bar=True, on_epoch=True)

        logger.info(
            f"Epoch {self.current_epoch} | PSNR=%.2f | CER=%.3f | SeqAcc=%.1f%%",
            self.trainer.logged_metrics.get("val/psnr", 0),
            mean_cer,
            seq_acc * 100,
        )
        # Clear accumulators
        self._val_preds.clear()
        self._val_gts.clear()

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = self.hparams.weight_decay
        
        # Train restoration + refinement; PARSeq is frozen
        trainable_params = list(self.restoration_model.parameters()) + \
                           list(self.refinement.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

        max_epochs  = self.hparams.max_epochs
        warmup_eps  = self.hparams.warmup_epochs
        min_lr_frac = 1e-6 / max(lr, 1e-10)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_eps:
                return (epoch + 1) / max(warmup_eps, 1)
            progress = (epoch - warmup_eps) / max(max_epochs - warmup_eps, 1)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_frac + (1.0 - min_lr_frac) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @classmethod
    def from_checkpoint(
        cls,
        model_name: str,
        parseq: nn.Module,
        device_str: str = "cuda",
        experiments_dir: str = "outputs/experiments",
        **kwargs,
    ) -> "JointOptimizationLightningModule":
        """Auto-detect best Phase 4 checkpoint and initialize module."""
        ckpt = _find_best_checkpoint(model_name, experiments_dir)
        if ckpt:
            logger.info(f"Auto-detected best Phase 4 checkpoint: {ckpt}")
        else:
            logger.warning(f"No Phase 4 checkpoint found for {model_name}. Starting from scratch.")
        return cls(
            model_name=model_name,
            parseq=parseq,
            device_str=device_str,
            pretrained_ckpt=str(ckpt) if ckpt else None,
            **kwargs,
        )
