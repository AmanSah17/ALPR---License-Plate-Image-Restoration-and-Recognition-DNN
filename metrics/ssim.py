"""
Structural Similarity (SSIM) for restoration evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Compute mean SSIM over channels.

    Args:
        prediction: ``(C, H, W)`` float tensor.
        target: ``(C, H, W)`` float tensor.
        window_size: Gaussian window size (odd).
        max_val: Dynamic range.

    Returns:
        SSIM in [0, 1] (higher is better).
    """
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    c = prediction.shape[1]
    window = _gaussian_window(window_size, c, prediction.device, prediction.dtype)
    ssim_vals = []
    for ch in range(c):
        s = _ssim_single(prediction[:, ch : ch + 1], target[:, ch : ch + 1], window, max_val)
        ssim_vals.append(s)
    return float(torch.stack(ssim_vals).mean().item())


def _gaussian_window(size: int, channels: int, device, dtype):
    sigma = 1.5
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, size, size).contiguous()


def _ssim_single(x, y, window, max_val: float, c1: float = 0.01**2, c2: float = 0.03**2):
    mu_x = F.conv2d(x, window, padding=window.shape[-1] // 2, groups=1)
    mu_y = F.conv2d(y, window, padding=window.shape[-1] // 2, groups=1)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_x = F.conv2d(x * x, window, padding=window.shape[-1] // 2, groups=1) - mu_x2
    sigma_y = F.conv2d(y * y, window, padding=window.shape[-1] // 2, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window.shape[-1] // 2, groups=1) - mu_xy
    c1, c2 = c1 * max_val**2, c2 * max_val**2
    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim.mean()
