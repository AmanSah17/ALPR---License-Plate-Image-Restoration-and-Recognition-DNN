import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import torchvision.transforms.functional as TF

class OCRSupervisor(nn.Module):
    """
    Abstract interface for a differentiable OCR engine used to supervise
    image restoration.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
    def get_encoder(self) -> Optional[nn.Module]:
        """Return the underlying image encoder module if perceptual loss is supported, else None."""
        raise NotImplementedError
        
    def encode_labels(self, texts: List[str], target_len: int) -> torch.Tensor:
        """
        Encode ground truth text strings into target token indices.
        Returns tensor of shape (B, target_len).
        """
        raise NotImplementedError

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        """
        Decode model output logits back into strings for validation metrics.
        """
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            images: Tensor of shape (B, 3, H, W) in range [0, 1].
        Returns:
            Logits tensor of shape (B, T, V).
        """
        raise NotImplementedError


class PARSeqSupervisor(OCRSupervisor):
    """
    Wraps the baudm/parseq model as a differentiable supervisor.
    """
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True).to(device)
        self.parseq.eval()
        for p in self.parseq.parameters():
            p.requires_grad_(False)
            
    def get_encoder(self) -> Optional[nn.Module]:
        if hasattr(self.parseq, 'encoder'):
            return self.parseq.encoder
        elif hasattr(self.parseq, 'model') and hasattr(self.parseq.model, 'encoder'):
            return self.parseq.model.encoder
        return None

    def encode_labels(self, texts: List[str], target_len: int) -> torch.Tensor:
        encoded = self.parseq.tokenizer.encode(texts)
        B = len(encoded)
        padded = torch.full((B, target_len), self.parseq.tokenizer.pad_id, dtype=torch.long, device=self.device)
        for i, t in enumerate(encoded):
            t_stripped = t[1:] # Strip BOS token
            n = min(t_stripped.shape[0], target_len)
            padded[i, :n] = t_stripped[:n]
        return padded

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        probs = logits.softmax(-1)
        labels, _ = self.parseq.tokenizer.decode(probs)
        return labels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # PARSeq expects 32x128 in [-1, 1]
        ocr_input = TF.resize(images, (32, 128), antialias=True) * 2.0 - 1.0
        # Forward in FP32 to prevent NaN overflow in ViT
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = self.parseq(ocr_input.float())
        return logits


class FastPlateOCRValidator:
    """
    Wraps fast-plate-ocr for validation inference only.
    Because it uses ONNX Runtime, it cannot be used for PyTorch backprop gradients.
    """
    def __init__(self, device: torch.device):
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            # FastPlateOCR handles device internally if onnxruntime-gpu is available
            self.recognizer = LicensePlateRecognizer(
                hub_ocr_model="global-plates-mobile-vit-v2-model",
                device="cuda" if device.type == "cuda" else "cpu"
            )
        except ImportError:
            import logging
            logging.getLogger(__name__).warning("fast-plate-ocr not installed. FastPlateOCRValidator disabled.")
            self.recognizer = None

    def decode_images(self, images: torch.Tensor) -> List[str]:
        """
        Takes (B, 3, H, W) PyTorch tensors in [0, 1], converts to numpy BGR [0, 255]
        and runs fast-plate-ocr inference.
        Returns list of predicted strings.
        """
        if self.recognizer is None:
            return [""] * images.shape[0]

        import numpy as np
        # Convert (B, 3, H, W) [0, 1] RGB -> (B, H, W, 3) [0, 255] RGB
        imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        
        results = []
        for img in imgs_np:
            # OpenCV BGR format is required by fast-plate-ocr
            img_bgr = img[:, :, ::-1]
            try:
                # FastPlateOCR outputs a tuple: (text, confidence)
                text, conf = self.recognizer.run(img_bgr)
                results.append(text)
            except Exception as e:
                results.append("")
        return results
