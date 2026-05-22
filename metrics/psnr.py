"""
Peak Signal-to-Noise Ratio (PSNR) for restoration evaluation.
"""

from __future__ import annotations

import torch


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """
    Compute PSNR between two images.

    Args:
        prediction: ``(C, H, W)`` or ``(B, C, H, W)`` in [0, max_val].
        target: Same shape as prediction.
        max_val: Peak value (1.0 for normalized tensors).
        eps: Numerical stability.

    Returns:
        PSNR in dB (higher is better).
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    mse = torch.mean((prediction - target) ** 2).item()
    if mse < eps:
        return 100.0
    import math

    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)
