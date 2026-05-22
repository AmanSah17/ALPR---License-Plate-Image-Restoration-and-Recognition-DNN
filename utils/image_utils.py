"""
Image I/O and tensor conversion utilities (used across dataset and visualization).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_image_rgb(path: PathLike) -> np.ndarray:
    """
    Load an image as RGB uint8 numpy array.

    Args:
        path: Image file path.

    Returns:
        Array of shape ``(H, W, 3)`` with dtype ``uint8``.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If image cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def save_image_rgb(array: np.ndarray, path: PathLike) -> Path:
    """
    Save RGB or grayscale array to PNG.

    Args:
        array: ``(H, W)`` or ``(H, W, 3)`` uint8 array.
        path: Output path.

    Returns:
        Written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        mode = "L"
    elif array.ndim == 3 and array.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported array shape for save: {array.shape}")
    Image.fromarray(array.astype(np.uint8), mode=mode).save(path)
    return path


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    Normalize uint8/float image to zero-centered range.

    Args:
        image: ``(H, W, 3)`` array.
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        Float32 normalized array.
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (img - mean_arr) / std_arr


def denormalize_image(
    tensor_like: np.ndarray,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Inverse of ``normalize_image``; returns uint8 RGB."""
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = tensor_like * std_arr + mean_arr
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def ensure_hwc(image: np.ndarray) -> np.ndarray:
    """Ensure image is HWC; convert CHW if detected."""
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[0] < image.shape[-1]:
        return np.transpose(image, (1, 2, 0))
    return image


def stack_frames(frames: List[np.ndarray]) -> np.ndarray:
    """
    Stack list of HWC frames into ``(T, H, W, C)`` array.

    Raises:
        ValueError: If shapes differ.
    """
    if not frames:
        raise ValueError("Cannot stack empty frame list.")
    ref = frames[0].shape
    for idx, frame in enumerate(frames[1:], start=1):
        if frame.shape != ref:
            raise ValueError(f"Frame {idx} shape {frame.shape} != reference {ref}")
    return np.stack(frames, axis=0)
