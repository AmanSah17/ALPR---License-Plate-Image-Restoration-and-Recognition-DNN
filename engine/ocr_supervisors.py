from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


def normalize_ocr_backend_name(name: Optional[str]) -> str:
    """Normalize OCR backend aliases into a stable internal name."""
    if not name:
        return "none"
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "fastplateocr": "fast_plate_ocr",
        "fpo": "fast_plate_ocr",
        "noop": "none",
        "disabled": "none",
        "off": "none",
    }
    return aliases.get(normalized, normalized)


class OCRSupervisor(nn.Module):
    """
    Differentiable OCR interface used by the joint training loss.

    Backends that cannot support backprop can still implement this interface,
    but should expose ``supports_training_loss = False`` so the loss can skip
    OCR-specific terms cleanly.
    """

    backend_name: str = "base"
    supports_training_loss: bool = True

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    @property
    def pad_id(self) -> int:
        return -100

    def get_encoder(self) -> Optional[nn.Module]:
        """Return the underlying visual encoder if perceptual loss is supported."""
        raise NotImplementedError

    def encode_labels(self, texts: List[str], target_len: int) -> torch.Tensor:
        """Encode ground-truth text strings into token targets of shape ``(B, T)``."""
        raise NotImplementedError

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        """Decode model output logits back into strings."""
        raise NotImplementedError

    def decode_logits_with_confidence(self, logits: torch.Tensor) -> Tuple[List[str], List[float]]:
        """Decode logits into strings plus a scalar confidence per sample."""
        labels = self.decode_logits(logits)
        return labels, [0.0] * len(labels)

    def token_id_for_char(self, char: str) -> Optional[int]:
        """Resolve a single alphanumeric character to the backend token id if possible."""
        return None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape ``(B, 3, H, W)`` in range ``[0, 1]``.
        Returns:
            Logits tensor of shape ``(B, T, V)``.
        """
        raise NotImplementedError


class NoOpOCRSupervisor(OCRSupervisor):
    """Training-safe placeholder used when OCR supervision is intentionally disabled."""

    backend_name = "none"
    supports_training_loss = False

    def get_encoder(self) -> Optional[nn.Module]:
        return None

    def encode_labels(self, texts: List[str], target_len: int) -> torch.Tensor:
        return torch.full((len(texts), target_len), -100, dtype=torch.long, device=self.device)

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        return [""] * logits.shape[0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("OCR supervision is disabled for the 'none' backend.")


class PARSeqSupervisor(OCRSupervisor):
    """Differentiable OCR supervisor backed by ``baudm/parseq``."""

    backend_name = "parseq"
    supports_training_loss = True

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).to(device)
        self.parseq.eval()
        for p in self.parseq.parameters():
            p.requires_grad_(False)

    @property
    def pad_id(self) -> int:
        return int(self.parseq.tokenizer.pad_id)

    def get_encoder(self) -> Optional[nn.Module]:
        if hasattr(self.parseq, "encoder"):
            return self.parseq.encoder
        if hasattr(self.parseq, "model") and hasattr(self.parseq.model, "encoder"):
            return self.parseq.model.encoder
        return None

    def encode_labels(self, texts: List[str], target_len: int) -> torch.Tensor:
        encoded = self.parseq.tokenizer.encode(texts).to(self.device)
        if encoded.dim() != 2:
            raise ValueError(f"Unexpected PARSeq token shape: {tuple(encoded.shape)}")

        batch_size = encoded.shape[0]
        padded = torch.full((batch_size, target_len), self.pad_id, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            tokens = encoded[i]
            eos_positions = (tokens == self.parseq.tokenizer.eos_id).nonzero(as_tuple=False)
            eos_idx = int(eos_positions[0].item()) if eos_positions.numel() else tokens.shape[0]
            stripped = tokens[1:eos_idx]
            n = min(int(stripped.shape[0]), target_len)
            if n > 0:
                padded[i, :n] = stripped[:n]
        return padded

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        probs = logits.softmax(-1)
        labels, _ = self.parseq.tokenizer.decode(probs)
        return labels

    def decode_logits_with_confidence(self, logits: torch.Tensor) -> Tuple[List[str], List[float]]:
        probs = logits.softmax(-1)
        labels, seq_probs = self.parseq.tokenizer.decode(probs)
        confidences = [float(p.float().mean().item()) if p.numel() else 0.0 for p in seq_probs]
        return labels, confidences

    def token_id_for_char(self, char: str) -> Optional[int]:
        if not char:
            return None
        tokens = self.parseq.tokenizer.encode([char])
        if tokens.dim() != 2 or tokens.shape[1] < 2:
            return None
        return int(tokens[0, 1].item())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        ocr_input = TF.resize(images, (32, 128), antialias=True) * 2.0 - 1.0
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = self.parseq(ocr_input.float())
        return logits


class OCRValidator:
    """Inference-only OCR interface used for validation, benchmarking, and deployment."""

    backend_name: str = "base"

    def __init__(self, device: torch.device):
        self.device = device

    def decode_images(self, images: torch.Tensor) -> Tuple[List[str], List[float]]:
        raise NotImplementedError


class NoOpOCRValidator(OCRValidator):
    backend_name = "none"

    def decode_images(self, images: torch.Tensor) -> Tuple[List[str], List[float]]:
        batch = int(images.shape[0])
        return [""] * batch, [0.0] * batch


class PARSeqValidator(OCRValidator):
    """Inference adapter for PARSeq so training/eval backends can share one API."""

    backend_name = "parseq"

    def __init__(self, device: torch.device, supervisor: Optional[PARSeqSupervisor] = None):
        super().__init__(device)
        self.supervisor = supervisor or PARSeqSupervisor(device)

    def decode_images(self, images: torch.Tensor) -> Tuple[List[str], List[float]]:
        with torch.no_grad():
            logits = self.supervisor(images.to(self.device))
        return self.supervisor.decode_logits_with_confidence(logits)


class FastPlateOCRValidator(OCRValidator):
    """
    ``fast-plate-ocr`` adapter for validation and live inference.

    This backend is ONNX Runtime based and therefore not differentiable.
    """

    backend_name = "fast_plate_ocr"

    def __init__(self, device: torch.device, model_name: str = "global-plates-mobile-vit-v2-model"):
        super().__init__(device)
        self.recognizer = None
        self.model_name = model_name
        try:
            from fast_plate_ocr import LicensePlateRecognizer

            self.recognizer = LicensePlateRecognizer(
                hub_ocr_model=model_name,
                device="cuda" if device.type == "cuda" else "cpu",
            )
        except ImportError:
            logger.warning("fast-plate-ocr is not installed. FastPlateOCRValidator disabled.")
        except Exception as exc:
            logger.warning("Failed to initialize fast-plate-ocr validator: %s", exc)

    def _prepare_numpy_batch(self, images: torch.Tensor):
        import numpy as np

        rgb = (images.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        if self.recognizer is None:
            return rgb

        color_mode = getattr(self.recognizer.config, "image_color_mode", "rgb")
        if str(color_mode).lower() == "grayscale":
            grayscale = []
            for img in rgb:
                gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
                grayscale.append(gray)
            return grayscale
        return rgb

    def decode_images(self, images: torch.Tensor) -> Tuple[List[str], List[float]]:
        batch = int(images.shape[0])
        if self.recognizer is None:
            return [""] * batch, [0.0] * batch

        prepared = self._prepare_numpy_batch(images)
        try:
            predictions = self.recognizer.run(prepared, return_confidence=True)
        except Exception as exc:
            logger.warning("fast-plate-ocr inference failed: %s", exc)
            return [""] * batch, [0.0] * batch

        labels: List[str] = []
        confidences: List[float] = []
        for pred in predictions:
            labels.append(str(getattr(pred, "plate", "")))
            char_probs = getattr(pred, "char_probs", None)
            if char_probs is None or len(char_probs) == 0:
                confidences.append(0.0)
            else:
                confidences.append(float(torch.as_tensor(char_probs).float().mean().item()))
        return labels, confidences


def build_ocr_supervisor(backend: str, device: torch.device) -> OCRSupervisor:
    """Factory for differentiable OCR supervisors used by the joint loss."""
    normalized = normalize_ocr_backend_name(backend)
    if normalized == "parseq":
        return PARSeqSupervisor(device)
    if normalized == "none":
        return NoOpOCRSupervisor(device)
    if normalized in {"fast_plate_ocr", "svtrv2", "lcofl"}:
        logger.warning(
            "OCR backend '%s' does not expose a differentiable training supervisor here. "
            "Falling back to 'none' for OCR loss.",
            normalized,
        )
        return NoOpOCRSupervisor(device)
    raise ValueError(f"Unsupported OCR supervisor backend: {backend}")


def build_ocr_validator(
    backend: str,
    device: torch.device,
    supervisor: Optional[OCRSupervisor] = None,
) -> OCRValidator:
    """Factory for inference OCR backends used in validation and deployment."""
    normalized = normalize_ocr_backend_name(backend)
    if normalized == "parseq":
        if isinstance(supervisor, PARSeqSupervisor):
            return PARSeqValidator(device, supervisor=supervisor)
        return PARSeqValidator(device)
    if normalized == "fast_plate_ocr":
        return FastPlateOCRValidator(device)
    if normalized == "none":
        return NoOpOCRValidator(device)
    raise ValueError(f"Unsupported OCR validator backend: {backend}")


def list_supported_ocr_backends() -> Sequence[str]:
    return ("parseq", "fast_plate_ocr", "none")
