"""
GPU utilities optimized for low-VRAM CUDA devices (e.g. GTX 1650 4GB).
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.cuda.amp import autocast
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    autocast = None  # type: ignore[assignment,misc]


@dataclass
class GPUMemoryStats:
    """Snapshot of GPU memory usage in megabytes."""

    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    device_name: str
    device_index: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "max_allocated_mb": self.max_allocated_mb,
            "device_index": float(self.device_index),
        }


@dataclass
class GPUConfig:
    """Low-VRAM oriented GPU runtime configuration."""

    device: str = "auto"
    precision: str = "fp16"
    low_vram_mode: bool = True
    max_split_size_mb: int = 128
    allow_tf32: bool = False
    empty_cache_interval: int = 10


class GPUManager:
    """
    Device selection, mixed-precision contexts, and memory profiling.

    Designed for 4GB GPUs: enables memory splitting, cache clearing,
    and conservative autocast defaults.
    """

    SUPPORTED_PRECISION = {"fp32", "fp16", "bf16"}

    def __init__(self, config: Optional[GPUConfig] = None) -> None:
        self.config = config or GPUConfig()
        self._apply_low_vram_env()
        self._device = self._resolve_device(self.config.device)
        self._step_counter = 0

    def _apply_low_vram_env(self) -> None:
        """Set CUDA allocator env vars before first CUDA allocation."""
        if not self.config.low_vram_mode:
            return
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            f"max_split_size_mb:{self.config.max_split_size_mb}",
        )

    def _resolve_device(self, device: str) -> "torch.device":
        if torch is None:
            raise RuntimeError("PyTorch is required for GPUManager.")
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
            logger.warning("CUDA not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    @property
    def device(self) -> "torch.device":
        """Active torch device."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        return self._device.type == "cuda"

    def get_device(self) -> "torch.device":
        """Return the resolved device (alias for ``device`` property)."""
        return self._device

    def empty_cache(self) -> None:
        """Release unused cached GPU memory."""
        if torch is None or not self.is_cuda:
            return
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self._device)

    def maybe_empty_cache(self) -> None:
        """Periodically clear cache based on ``empty_cache_interval``."""
        self._step_counter += 1
        if (
            self.config.empty_cache_interval > 0
            and self._step_counter % self.config.empty_cache_interval == 0
        ):
            self.empty_cache()

    def memory_stats(self) -> GPUMemoryStats:
        """
        Collect current GPU memory statistics.

        Returns:
            ``GPUMemoryStats``; zeros on CPU.
        """
        if torch is None or not self.is_cuda:
            return GPUMemoryStats(0.0, 0.0, 0.0, "cpu", -1)
        idx = self._device.index or 0
        allocated = torch.cuda.memory_allocated(idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
        name = torch.cuda.get_device_name(idx)
        return GPUMemoryStats(allocated, reserved, max_allocated, name, idx)

    def reset_peak_memory(self) -> None:
        """Reset peak memory counters."""
        if torch is not None and self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)

    def autocast_context(self, enabled: Optional[bool] = None) -> Any:
        """
        Return a mixed-precision autocast context manager.

        Args:
            enabled: Override; if ``None``, inferred from ``precision`` config.

        Returns:
            ``torch.cuda.amp.autocast`` or nullcontext on CPU/fp32.
        """
        if torch is None:
            raise RuntimeError("PyTorch required for autocast.")
        use_amp = enabled if enabled is not None else self.config.precision in {"fp16", "bf16"}
        if not self.is_cuda or not use_amp:
            return contextlib.nullcontext()
        dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        # Compatible with torch.cuda.amp (legacy) and torch.amp (2.x+)
        try:
            return torch.amp.autocast("cuda", dtype=dtype)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return autocast(dtype=dtype)

    def to_device(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Move tensor to the managed device."""
        return tensor.to(self._device, non_blocking=self.is_cuda)

    def move_module(self, module: "nn.Module") -> "nn.Module":
        """Move ``nn.Module`` to managed device."""
        return module.to(self._device)

    def recommended_batch_size(self, per_sample_mb: float, safety_factor: float = 0.85) -> int:
        """
        Estimate a safe batch size from per-sample VRAM usage.

        Args:
            per_sample_mb: Estimated MB per sample at current resolution.
            safety_factor: Fraction of free memory to use.

        Returns:
            At least 1.
        """
        if not self.is_cuda or torch is None:
            return 1
        idx = self._device.index or 0
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
        stats = self.memory_stats()
        free = max(total - stats.allocated_mb, 0.0) * safety_factor
        if per_sample_mb <= 0:
            return 1
        return max(1, int(free // per_sample_mb))

    def log_status(self) -> None:
        """Log device and memory information."""
        if not self.is_cuda:
            logger.info("Running on CPU.")
            return
        stats = self.memory_stats()
        logger.info(
            "GPU [%s:%d] %s | alloc=%.1f MB reserved=%.1f MB peak=%.1f MB",
            stats.device_name,
            stats.device_index,
            self.config.precision,
            stats.allocated_mb,
            stats.reserved_mb,
            stats.max_allocated_mb,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "GPUManager":
        """Instantiate from OmegaConf ``hardware`` or ``gpu`` section."""
        hw = cfg.get("hardware", cfg.get("gpu", cfg)) if hasattr(cfg, "get") else cfg
        return cls(
            GPUConfig(
                device=str(hw.get("device", "auto")),
                precision=str(hw.get("precision", "fp16")),
                low_vram_mode=bool(hw.get("low_vram_mode", True)),
                max_split_size_mb=int(hw.get("max_split_size_mb", 128)),
                allow_tf32=bool(hw.get("allow_tf32", False)),
                empty_cache_interval=int(hw.get("empty_cache_interval", 10)),
            )
        )


def get_device(device: str = "auto") -> "torch.device":
    """Resolve a torch device string."""
    return GPUManager(GPUConfig(device=device)).get_device()
