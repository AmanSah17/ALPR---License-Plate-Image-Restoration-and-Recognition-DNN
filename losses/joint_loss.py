"""
Phase 7: Joint Differentiable Loss combining:
# Improvements from research (2024):
# - Sigmoid warmup annealing instead of linear (smoother)
# - L2-normalized feature loss (more stable than raw MSE)
# - VRAM-aware feature loss gating (skip when <500MB free)
- Pixel-level L1/SSIM restoration loss
- OCR Cross-Entropy loss from PARSeq logits
- Perceptual feature matching via PARSeq encoder intermediate layers

Loss formulation:
    L_total = λ_pixel * L_pixel(restored, pseudo_gt)
            + λ_ocr   * L_ce(parseq_logits, gt_tokens)
            + λ_feat  * L_feat(parseq_feats(restored), parseq_feats(pseudo_gt))

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
    ocr_weight: float = 0.1         # PARSeq cross-entropy loss weight
    feat_weight: float = 0.05       # Perceptual feature matching weight
    ssim_weight: float = 0.1        # SSIM component of pixel loss
    
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


class ParseqOCRLoss(nn.Module):
    """
    Differentiable OCR loss using frozen PARSeq logits.
    
    PARSeq outputs (B, T, V) logits where:
        T = max sequence length (padded)
        V = vocabulary size (charset + special tokens)
    
    We compute CrossEntropyLoss over the character positions
    so that gradients flow BACKWARD into the restoration model.
    """
    def __init__(self, parseq: nn.Module, device: torch.device):
        super().__init__()
        self.parseq = parseq
        self.device = device
        # Fully freeze PARSeq - it's a fixed OCR oracle
        for p in self.parseq.parameters():
            p.requires_grad_(False)
        self.parseq.eval()
    
    def encode_labels(self, texts: list, target_len: int) -> torch.Tensor:
        """Encode text strings to PARSeq token targets, padded to target_len.
        
        tokenizer.encode(texts) returns [BOS, chars..., EOS].
        PARSeq logits predict [chars..., EOS, PAD...].
        We must strip the BOS token and pad the rest to target_len.
        """
        encoded = self.parseq.tokenizer.encode(texts)  # list of 1D tensors
        B = len(encoded)
        padded = torch.full(
            (B, target_len),
            self.parseq.tokenizer.pad_id,
            dtype=torch.long,
        )
        for i, t in enumerate(encoded):
            # Strip BOS token (t[0]) because logits don't predict BOS
            t_stripped = t[1:]
            n = min(t_stripped.shape[0], target_len)
            padded[i, :n] = t_stripped[:n]
        return padded.to(self.device)
    
    def forward(
        self,
        refined_images: torch.Tensor,   # (B, 3, H, W) in [0, 1]
        gt_texts: list,                  # list of ground truth strings
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            refined_images: Batch of restored & refined images.
            gt_texts: Ground truth plate text strings.
        Returns:
            (ocr_loss, logits) where logits can be decoded for metrics.
        """
        import torchvision.transforms.functional as TF
        
        # PARSeq expects (32, 128) normalized to [-1, 1]
        ocr_input = TF.resize(refined_images, (32, 128), antialias=True)
        ocr_input = ocr_input * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Forward through frozen PARSeq
        logits = self.parseq(ocr_input)  # (B, T, V)
        B, T, V = logits.shape
        
        # Encode ground truth text to token indices (B, T) — padded to logit length
        # BOS token is stripped automatically. Valid classes are 0..(V-1) and pad_id.
        targets = self.encode_labels(gt_texts, target_len=T)  # (B, T)
        
        # Compute cross-entropy over sequence positions
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=self.parseq.tokenizer.pad_id,
        )
        return loss, logits


class PerceptualOCRFeatureLoss(nn.Module):
    """
    Feature-level perceptual loss using PARSeq's visual encoder.
    Forces the restored image to produce similar CNN features as the pseudo-GT.
    
    This is analogous to LPIPS but uses the OCR encoder instead of VGG.
    """
    def __init__(self, parseq: nn.Module):
        super().__init__()
        self._encoder = None
        self._init_encoder(parseq)
        if self._encoder is not None:
            for p in self._encoder.parameters():
                p.requires_grad_(False)
    
    def _init_encoder(self, parseq: nn.Module):
        """Extract PARSeq's image encoder sub-module."""
        # PARSeq has an encoder attribute (ViT-based or ResNet-based)
        if hasattr(parseq, 'encoder'):
            self._encoder = parseq.encoder
        elif hasattr(parseq, 'model') and hasattr(parseq.model, 'encoder'):
            self._encoder = parseq.model.encoder
        else:
            logger.warning("Could not extract PARSeq encoder for feature loss. Disabling.")
    
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
                feat_target = self._encoder(tgt_r)
            feat_pred = self._encoder(pred_r)
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
    - OCR loss (PARSeq CE)
    - Perceptual feature matching
    - Loss annealing schedule
    """
    
    def __init__(
        self,
        parseq: nn.Module,
        device: torch.device,
        weights: Optional[JointLossWeights] = None,
    ):
        super().__init__()
        self.weights = weights or JointLossWeights()
        self.device = device
        
        # Component losses
        self.ssim_loss = SSIMLoss(channels=3).to(device)
        self.ocr_loss  = ParseqOCRLoss(parseq, device)
        self.feat_loss = PerceptualOCRFeatureLoss(parseq)
        
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
    def from_config(cls, parseq: nn.Module, device: torch.device, cfg=None) -> "JointLoss":
        """Build JointLoss from config dict or with defaults."""
        w = JointLossWeights()
        if cfg is not None:
            jcfg = getattr(cfg, "joint_loss", cfg.get("joint_loss", {})) if hasattr(cfg, "get") else {}
            w.pixel_weight      = float(jcfg.get("pixel_weight", w.pixel_weight))
            w.ocr_weight        = float(jcfg.get("ocr_weight", w.ocr_weight))
            w.feat_weight       = float(jcfg.get("feat_weight", w.feat_weight))
            w.ssim_weight       = float(jcfg.get("ssim_weight", w.ssim_weight))
            w.ocr_warmup_epochs = int(jcfg.get("ocr_warmup_epochs", w.ocr_warmup_epochs))
            w.feat_warmup_epochs = int(jcfg.get("feat_warmup_epochs", w.feat_warmup_epochs))
        return cls(parseq=parseq, device=device, weights=w)
