"""
PyTorch Lightning module for SwinIRUNetHybrid restoration training (Phase 5.1).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from metrics.psnr import compute_psnr
from metrics.ssim import compute_ssim
from models.losses.restoration_losses import RestorationLoss
from models.restoration.swinir_unet import SwinIRUNetHybrid

logger = logging.getLogger(__name__)


class RestorationLightningModule(pl.LightningModule):
    """
    Lightning wrapper with MLflow-loggable training/validation metrics.

    Logs per step: ``train/loss``, ``train/l1``, ``train/ssim``, ``train/lpips``
    Logs per epoch: ``val/psnr``, ``val/ssim``, ``val/loss``, ``val/psnr_gain_vs_center``
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.model = SwinIRUNetHybrid.from_config(cfg)
        self.criterion = RestorationLoss.from_config(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        inp = batch["input"]
        tgt = batch["target"]
        if inp.dim() == 3:
            inp = inp.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
        pred = self(inp)
        loss, components = self.criterion(pred, tgt)

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

        if stage == "val":
            pred0 = pred[0].detach().cpu()
            tgt0 = tgt[0].detach().cpu()
            inp0 = inp[0].detach().cpu()
            psnr = compute_psnr(pred0, tgt0)
            ssim = compute_ssim(pred0, tgt0)
            psnr_base = compute_psnr(inp0, tgt0)
            self.log("val/psnr", psnr, prog_bar=True, on_epoch=True)
            self.log("val_psnr", psnr, on_epoch=True)  # filesystem-safe alias for checkpoints
            self.log("val/ssim", ssim, on_epoch=True)
            self.log("val/psnr_gain_vs_center", psnr - psnr_base, on_epoch=True)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt_cfg = self.cfg.optimizer
        lr = float(opt_cfg.get("lr", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 1e-4))
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=betas,
        )

        sched_cfg = self.cfg.scheduler
        max_epochs = int(self.cfg.trainer.get("max_epochs", 100))
        warmup = int(sched_cfg.get("warmup_epochs", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = (epoch - warmup) / float(max(1, max_epochs - warmup))
            import math

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr / lr + (1.0 - min_lr / lr) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
