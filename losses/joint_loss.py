"""
Phase 7: Joint Differentiable Loss combining:
# Improvements from research (2024):
# - Sigmoid warmup annealing instead of linear (smoother)
# - L2-normalized feature loss (more stable than raw MSE)
# - VRAM-aware feature loss gating (skip when <500MB free)
- Pixel-level L1/SSIM restoration loss
- OCR supervision loss from a modular differentiable OCR backend
- Perceptual feature matching via the OCR encoder intermediate layers

Loss formulation:
    L_total = λ_pixel * L_pixel(restored, pseudo_gt)
            + λ_ocr   * L_ocr(ocr_logits, gt_tokens)
            + λ_feat  * L_feat(ocr_feats(restored), ocr_feats(pseudo_gt))

Copyright (c) 2024 Aman Sah (amansah1717@gmail.com)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class JointLossWeights:
    """Configuration for joint loss weighting with cosine warmup annealing."""
    pixel_weight: float = 1.0       # L1 + SSIM restoration loss weight
    ocr_weight: float = 0.1         # OCR supervision loss weight
    feat_weight: float = 0.05       # Perceptual feature matching weight
    ssim_weight: float = 0.1        # SSIM component of pixel loss
    ocr_loss_type: str = "lcofl"    # ce | lcofl
    ocr_focal_gamma: float = 2.0
    
    # Annealing: ramp OCR loss in from 0 over N epochs so pixel loss
    # stabilises the restoration network first before OCR gradient kicks in.
    ocr_warmup_epochs: int = 5
    feat_warmup_epochs: int = 10
    
    def ocr_scale(self, epoch: int) -> float:
        """Sigmoid annealing: smooth S-curve warmup instead of linear."""
        if self.ocr_warmup_epochs <= 0:
            return 1.0
        x = (epoch - self.ocr_warmup_epochs / 2) / max(self.ocr_warmup_epochs / 6, 0.1)
        import math
        return min(1.0, 1.0 / (1.0 + math.exp(-x)))
    
    def feat_scale(self, epoch: int) -> float:
        """Sigmoid annealing for feature loss."""
        if self.feat_warmup_epochs <= 0:
            return 1.0
        x = (epoch - self.feat_warmup_epochs / 2) / max(self.feat_warmup_epochs / 6, 0.1)
        import math
        return min(1.0, 1.0 / (1.0 + math.exp(-x)))


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM)."""
    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        # Build fixed Gaussian kernel
        self.register_buffer("kernel", self._build_kernel(window_size, channels))
    
    def _build_kernel(self, window_size: int, channels: int) -> torch.Tensor:
        import math
        sigma = 1.5
        gauss = torch.FloatTensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss /= gauss.sum()
        kernel_2d = gauss.outer(gauss)
        return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to prevent FP16 overflow/underflow in variance calculation
        pred = pred.clamp(0, 1).float()
        target = target.clamp(0, 1).float()
        
        mu1 = F.conv2d(pred, self.kernel, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(target, self.kernel, padding=self.window_size // 2, groups=self.channels)
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, self.kernel, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12   = F.conv2d(pred * target, self.kernel, padding=self.window_size // 2, groups=self.channels) - mu1_mu2
        
        # Ensure variances are strictly positive
        sigma1_sq = torch.relu(sigma1_sq)
        sigma2_sq = torch.relu(sigma2_sq)
        
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return 1.0 - ssim_map.mean()


class ModularOCRLoss(nn.Module):
    """
    Differentiable OCR Loss using any OCRSupervisor.
    Computes LCOFL (Layout and Character Oriented Focal Loss) over the character logits.
    """
    _CONFUSION_GROUPS = (
        ("0", "O", "D", "Q"),
        ("1", "I", "L"),
        ("2", "Z"),
        ("5", "S"),
        ("6", "G"),
        ("8", "B"),
        ("7", "T"),
    )

    def __init__(
        self,
        supervisor: 'OCRSupervisor',
        device: torch.device,
        loss_type: str = "lcofl",
        gamma: float = 2.0,
    ):
        super().__init__()
        self.supervisor = supervisor
        self.device = device
        self.loss_type = loss_type.lower()
        self.gamma = gamma
        self.confusion_map = self._build_confusion_map()

    def _build_confusion_map(self) -> Dict[int, Tuple[int, ...]]:
        confusion_map: Dict[int, Tuple[int, ...]] = {}
        for group in self._CONFUSION_GROUPS:
            token_ids = []
            for ch in group:
                token_id = self.supervisor.token_id_for_char(ch)
                if token_id is not None:
                    token_ids.append(token_id)
            token_ids = sorted(set(token_ids))
            for token_id in token_ids:
                confused = tuple(other for other in token_ids if other != token_id)
                if confused:
                    confusion_map[token_id] = confused
        return confusion_map

    def _apply_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    def _apply_lcofl(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        positions: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Compute Layout and Character Oriented Focal Loss.
        logits: (N, V)
        targets: (N,)
        """
        ce_loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
        valid_mask = targets != ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        probs = logits.softmax(-1)
        target_probs = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_weight = (1.0 - target_probs) ** self.gamma

        confusion_weight = torch.ones_like(target_probs)
        for target_id, confused_ids in self.confusion_map.items():
            confused_tensor = torch.as_tensor(confused_ids, device=logits.device, dtype=torch.long)
            mask = valid_mask & (safe_targets == target_id)
            if not mask.any():
                continue
            confusion_pressure = probs[mask].index_select(1, confused_tensor).max(dim=1).values
            confusion_weight[mask] = 1.0 + confusion_pressure

        layout_weight = torch.ones_like(target_probs)
        if sequence_length > 2:
            edge_mask = (positions <= 1) | (positions >= sequence_length - 2)
            layout_weight = torch.where(edge_mask, layout_weight.new_tensor(1.1), layout_weight)

        total_weight = focal_weight * confusion_weight * layout_weight
        return (ce_loss[valid_mask] * total_weight[valid_mask]).mean()

    def forward(
        self,
        refined_images: torch.Tensor,   # (B, 3, H, W) in [0, 1]
        gt_texts: list,                  # list of ground truth strings
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not getattr(self.supervisor, "supports_training_loss", True):
            zero = torch.tensor(0.0, device=refined_images.device, requires_grad=True)
            return zero, None

        # 1. Forward through modular supervisor
        logits = self.supervisor(refined_images) # (B, T, V)
        B, T, V = logits.shape

        # 2. Encode targets using the supervisor's tokenizer
        targets = self.supervisor.encode_labels(gt_texts, target_len=T) # (B, T)

        # 3. OCR loss
        pad_id = getattr(self.supervisor, 'pad_id', -100)
        flat_logits = logits.reshape(B * T, V)
        flat_targets = targets.reshape(B * T)
        flat_positions = torch.arange(T, device=targets.device).unsqueeze(0).expand(B, T).reshape(B * T)

        if self.loss_type == "ce":
            loss = self._apply_cross_entropy(flat_logits, flat_targets, ignore_index=pad_id)
        else:
            loss = self._apply_lcofl(
                flat_logits,
                flat_targets,
                ignore_index=pad_id,
                positions=flat_positions,
                sequence_length=T,
            )
        return loss, logits


class PerceptualOCRFeatureLoss(nn.Module):
    """
    Feature-level perceptual loss using the OCR supervisor's visual encoder.
    Forces the restored image to produce similar CNN features as the pseudo-GT.
    
    This is analogous to LPIPS but uses the OCR encoder instead of VGG.
    """
    def __init__(self, encoder: Optional[nn.Module]):
        super().__init__()
        self._encoder = encoder
        if self._encoder is not None:
            for p in self._encoder.parameters():
                p.requires_grad_(False)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Restored images (B, 3, H, W) in [0, 1]
            target: Pseudo-GT images (B, 3, H, W) in [0, 1]
        Returns:
            Feature-level L2 loss.
        """
        if self._encoder is None:
            return torch.tensor(0.0, device=pred.device)
        
        import torchvision.transforms.functional as TF
        pred_r = TF.resize(pred, (32, 128), antialias=True) * 2.0 - 1.0
        tgt_r  = TF.resize(target, (32, 128), antialias=True) * 2.0 - 1.0
        
        try:
            with torch.no_grad():
                with torch.autocast(device_type=pred.device.type, enabled=False):
                    feat_target = self._encoder(tgt_r.float())
            with torch.autocast(device_type=pred.device.type, enabled=False):
                feat_pred = self._encoder(pred_r.float())
            # L1 on L2-normalized features — more stable than raw MSE (research 2024)
            feat_pred_n   = F.normalize(feat_pred.reshape(feat_pred.shape[0], -1), dim=-1)
            feat_target_n = F.normalize(feat_target.detach().reshape(feat_target.shape[0], -1), dim=-1)
            return F.l1_loss(feat_pred_n, feat_target_n)
        except Exception as e:
            logger.debug(f"Feature loss forward failed: {e}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class JointLoss(nn.Module):
    """
    Combined Phase 7 Joint Loss.
    
    Manages:
    - Pixel loss (L1 + SSIM)
    - OCR loss (CE or LCOFL-style)
    - Perceptual feature matching
    - Loss annealing schedule
    """
    
    def __init__(
        self,
        ocr_supervisor: 'OCRSupervisor',
        device: torch.device,
        weights: Optional[JointLossWeights] = None,
    ):
        super().__init__()
        self.weights = weights or JointLossWeights()
        self.device = device
        
        # Component losses
        self.ssim_loss = SSIMLoss(channels=3).to(device)
        self.ocr_loss  = ModularOCRLoss(
            ocr_supervisor,
            device,
            loss_type=self.weights.ocr_loss_type,
            gamma=self.weights.ocr_focal_gamma,
        )
        
        # Perceptual feature matching uses the underlying encoder from the supervisor
        encoder = ocr_supervisor.get_encoder()
        self.feat_loss = PerceptualOCRFeatureLoss(encoder)
        
        self._current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Call at start of each epoch to update annealing."""
        self._current_epoch = epoch
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gt_texts: list,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred: Restored images (B, 3, H, W) in [0, 1].
            target: Pseudo-GT images (B, 3, H, W) in [0, 1].
            gt_texts: List of ground truth text strings.
        Returns:
            (total_loss, component_dict)
        """
        components: Dict[str, torch.Tensor] = {}
        
        # 1. Pixel L1 loss
        l1 = F.l1_loss(pred, target)
        components["l1"] = l1
        
        # 2. SSIM loss
        ssim = self.ssim_loss(pred, target)
        components["ssim"] = ssim
        
        # 3. Pixel loss total
        pixel = l1 + self.weights.ssim_weight * ssim
        components["pixel"] = pixel
        
        # 4. OCR Cross-Entropy loss (with warmup annealing)
        ocr_scale = self.weights.ocr_scale(self._current_epoch)
        try:
            ocr, logits = self.ocr_loss(pred, gt_texts)
            components["ocr"] = ocr
            if logits is not None:
                components["ocr_logits"] = logits
        except Exception as e:
            logger.warning(f"OCR loss failed (epoch {self._current_epoch}): {e}")
            ocr = torch.tensor(0.0, device=self.device, requires_grad=True)
            components["ocr"] = ocr
        
        # 5. Perceptual feature loss (with warmup annealing + VRAM gate)
        feat_scale = self.weights.feat_scale(self._current_epoch)
        # Skip feature loss if VRAM is tight (<500MB free) to avoid OOM
        _compute_feat = True
        if torch.cuda.is_available():
            free_mem_gb = torch.cuda.mem_get_info()[0] / 1e9
            if free_mem_gb < 0.5:
                _compute_feat = False
                logger.debug(f"Skipping feature loss: only {free_mem_gb:.2f} GB VRAM free")
        try:
            feat = self.feat_loss(pred, target) if _compute_feat else torch.tensor(0.0, device=self.device)
            components["feat"] = feat
        except Exception as e:
            logger.debug(f"Feature loss failed: {e}")
            feat = torch.tensor(0.0, device=self.device)
            components["feat"] = feat
        
        # 6. Total loss
        total = (
            self.weights.pixel_weight * pixel
            + ocr_scale * self.weights.ocr_weight * ocr
            + feat_scale * self.weights.feat_weight * feat
        )
        components["total"] = total
        components["ocr_scale"] = torch.tensor(ocr_scale)
        components["feat_scale"] = torch.tensor(feat_scale)
        
        return total, components
    
    @classmethod
    def from_config(cls, ocr_supervisor: 'OCRSupervisor', device: torch.device, cfg=None) -> "JointLoss":
        """Build JointLoss from config dict or with defaults."""
        w = JointLossWeights()
        if cfg is not None:
            jcfg = getattr(cfg, "joint_loss", cfg.get("joint_loss", {})) if hasattr(cfg, "get") else {}
            w.pixel_weight      = float(jcfg.get("pixel_weight", w.pixel_weight))
            w.ocr_weight        = float(jcfg.get("ocr_weight", w.ocr_weight))
            w.feat_weight       = float(jcfg.get("feat_weight", w.feat_weight))
            w.ssim_weight       = float(jcfg.get("ssim_weight", w.ssim_weight))
            w.ocr_loss_type     = str(jcfg.get("ocr_loss_type", w.ocr_loss_type))
            w.ocr_focal_gamma   = float(jcfg.get("ocr_focal_gamma", w.ocr_focal_gamma))
            w.ocr_warmup_epochs = int(jcfg.get("ocr_warmup_epochs", w.ocr_warmup_epochs))
            w.feat_warmup_epochs = int(jcfg.get("feat_warmup_epochs", w.feat_warmup_epochs))
        return cls(ocr_supervisor=ocr_supervisor, device=device, weights=w)
