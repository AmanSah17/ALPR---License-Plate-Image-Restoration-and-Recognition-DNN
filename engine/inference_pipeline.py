from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from engine.ocr_supervisors import build_ocr_validator
from models.restoration.refinement import OCRRefinementHook

logger = logging.getLogger(__name__)


def _split_checkpoint_state(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a checkpoint into restoration and refinement state dicts."""
    restoration_state: Dict[str, torch.Tensor] = {}
    refinement_state: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("restoration_model."):
            restoration_state[key[len("restoration_model."):]] = value
        elif key.startswith("model."):
            restoration_state[key[len("model."):]] = value
        elif key.startswith("refinement."):
            refinement_state[key[len("refinement."):]] = value
        else:
            restoration_state[key] = value
    return restoration_state, refinement_state


class ALPRPipeline(nn.Module):
    """
    End-to-end ALPR inference pipeline for restoration plus OCR decoding.

    The repository currently assumes the input is already a plate crop or a
    plate-centric ROI. Full-frame plate detection is not part of this class.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        ocr_backend: str = "fast_plate_ocr",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.ocr_backend_name = ocr_backend

        logger.info("Initializing ALPR pipeline on %s...", self.device)

        self._load_restoration_stack()
        self.ocr_validator = build_ocr_validator(ocr_backend, self.device)

    def _load_restoration_stack(self) -> None:
        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        from models.model_registry import ModelRegistry, initialize_registry

        initialize_registry()
        self.restoration_model = ModelRegistry.build(self.model_name).to(self.device)

        raw = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state = raw.get("state_dict", raw)
        restoration_state, refinement_state = _split_checkpoint_state(state)

        missing, unexpected = self.restoration_model.load_state_dict(restoration_state, strict=False)
        logger.info(
            "Loaded restoration model '%s' (missing=%d, unexpected=%d).",
            self.model_name,
            len(missing),
            len(unexpected),
        )
        self.restoration_model.eval()

        refinement_enabled = bool(refinement_state)
        self.refinement = OCRRefinementHook(channels=3, enabled=refinement_enabled).to(self.device)
        if refinement_enabled:
            ref_missing, ref_unexpected = self.refinement.load_state_dict(refinement_state, strict=False)
            logger.info(
                "Loaded refinement hook from checkpoint (missing=%d, unexpected=%d).",
                len(ref_missing),
                len(ref_unexpected),
            )
        else:
            logger.info("No trained refinement hook found in checkpoint. Using restoration output directly.")
        self.refinement.eval()

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> dict:
        """
        Args:
            input_tensor: Tensor of shape ``(B, C, H, W)``.
        Returns:
            Dict with restored image, refined image, decoded text, confidence,
            and OCR backend metadata.
        """
        input_tensor = input_tensor.to(self.device)

        restored = self.restoration_model(input_tensor)
        refined = self.refinement(restored, restored)
        refined_clamped = torch.clamp(refined, 0.0, 1.0)

        text, confidence = self.ocr_validator.decode_images(refined_clamped)
        return {
            "restored": restored,
            "refined": refined_clamped,
            "text": text,
            "confidence": confidence,
            "ocr_backend": getattr(self.ocr_validator, "backend_name", self.ocr_backend_name),
        }
