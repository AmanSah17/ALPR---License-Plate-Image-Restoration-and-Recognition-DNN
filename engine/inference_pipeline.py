import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import logging
from pathlib import Path
from PIL import Image

from engine.composite_lightning import CompositeRestorationLightningModule
from models.restoration.refinement import OCRRefinementHook

logger = logging.getLogger(__name__)

class ALPRPipeline(nn.Module):
    """
    End-to-End ALPR Inference Pipeline.
    Integrates:
    - Custom Restoration (SwinIR, UNet, Spatiotemporal)
    - Phase 5: OCR-Aware Refinement (Edge emphasis)
    - Phase 6: PARSeq OCR Engine
    """
    def __init__(self, model_name: str, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        
        logger.info(f"Initializing ALPR Pipeline on {self.device}...")
        
        # 1. Load Restoration Model (Phase 1-4)
        self._load_restoration_model()
        
        # 2. Initialize OCR Refinement (Phase 5)
        self.refinement = OCRRefinementHook(channels=3, enabled=True).to(self.device)
        self.refinement.eval()
        
        # 3. Load PARSeq (Phase 6)
        self._load_parseq()

    def _load_restoration_model(self):
        """Dynamically load the best checkpoint."""
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        from models.model_registry import ModelRegistry, initialize_registry
        initialize_registry()
        self.restoration_model = ModelRegistry.build(self.model_name).to(self.device)
        
        # Load state dict manually
        raw = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state = raw.get("state_dict", raw)
        # Strip "model." prefix from Lightning
        state = { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }
        self.restoration_model.load_state_dict(state, strict=False)
        
        self.restoration_model.eval()
        logger.info(f"Successfully loaded restoration model: {self.model_name}")

    def _load_parseq(self):
        """Load the PARSeq OCR engine from torch hub."""
        logger.info("Loading PARSeq from torch.hub ('baudm/parseq')...")
        try:
            self.parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).to(self.device)
            self.parseq.eval()
            self.parseq_transform = T.Compose([
                T.Resize((32, 128), T.InterpolationMode.BICUBIC),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            logger.info("Successfully loaded PARSeq.")
        except Exception as e:
            logger.error(f"Failed to load PARSeq. Ensure 'parseq' is installed. Error: {e}")
            raise e

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> dict:
        """
        End-to-end forward pass.
        Args:
            input_tensor: Tensor of shape (B, C, H, W). For Spatiotemporal, C=15 (5 frames).
        Returns:
            dict containing:
                - 'restored': Raw restored image (B, 3, H, W)
                - 'refined': Edge-refined image (B, 3, H, W)
                - 'text': List of decoded strings (length B)
                - 'confidence': List of confidence scores
        """
        input_tensor = input_tensor.to(self.device)
        
        # 1. Restoration (Phase 1-4)
        restored = self.restoration_model(input_tensor)
        
        # 2. Refinement (Phase 5)
        # Refinement hook expects features and raw image. We use restored as both for post-processing.
        refined = self.refinement(restored, restored)
        
        # Clamp to valid image range
        refined_clamped = torch.clamp(refined, 0, 1)
        
        # 3. OCR (Phase 6)
        ocr_input = self.parseq_transform(refined_clamped)
        logits = self.parseq(ocr_input)
        
        # Greedy decoding
        pred = logits.softmax(-1)
        label, confidence = self.parseq.tokenizer.decode(pred)
        
        return {
            'restored': restored,
            'refined': refined,
            'text': label,
            'confidence': confidence
        }
