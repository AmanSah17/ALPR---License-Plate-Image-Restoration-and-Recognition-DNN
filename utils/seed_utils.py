"""
Reproducibility utilities: global seed fixing and deterministic backend flags.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover - torch required at runtime
    torch = None  # type: ignore[assignment]


@dataclass
class SeedConfig:
    """Container for reproducibility-related settings."""

    seed: int = 42
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    rank: int = 0  # distributed offset


class ReproducibilityManager:
    """
    Centralized reproducibility controller for experiments.

    Sets Python, NumPy, and PyTorch seeds and configures cuDNN behavior
    for deterministic training when requested.
    """

    def __init__(self, config: Optional[SeedConfig] = None) -> None:
        """
        Args:
            config: Seed configuration; uses defaults if ``None``.
        """
        self.config = config or SeedConfig()

    @property
    def effective_seed(self) -> int:
        """Per-rank seed offset for distributed training."""
        return self.config.seed + self.config.rank

    def apply(self) -> int:
        """
        Apply all reproducibility settings.

        Returns:
            The effective seed that was set.
        """
        seed = self.effective_seed
        self._set_python_seed(seed)
        self._set_numpy_seed(seed)
        self._set_torch_seed(seed)
        self._configure_cudnn()
        self._configure_tf32()
        logger.info(
            "Reproducibility applied (seed=%d, deterministic=%s, benchmark=%s)",
            seed,
            self.config.cudnn_deterministic,
            self.config.cudnn_benchmark,
        )
        return seed

    def _set_python_seed(self, seed: int) -> None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _set_numpy_seed(self, seed: int) -> None:
        np.random.seed(seed)

    def _set_torch_seed(self, seed: int) -> None:
        if torch is None:
            logger.warning("PyTorch not installed; skipping torch seed setup.")
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _configure_cudnn(self) -> None:
        if torch is None or not torch.backends.cudnn.is_available():
            return
        torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark

    def _configure_tf32(self) -> None:
        if torch is None:
            return
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = self.config.allow_tf32

    def state_dict(self) -> Dict[str, Any]:
        """Serialize reproducibility settings for experiment metadata."""
        return {
            "seed": self.config.seed,
            "effective_seed": self.effective_seed,
            "cudnn_deterministic": self.config.cudnn_deterministic,
            "cudnn_benchmark": self.config.cudnn_benchmark,
            "allow_tf32": self.config.allow_tf32,
            "rank": self.config.rank,
        }

    @classmethod
    def from_omegaconf(cls, cfg: Any, rank: int = 0) -> "ReproducibilityManager":
        """
        Build manager from a config node (e.g. ``cfg.reproducibility``).

        Args:
            cfg: Dict-like config with optional reproducibility section.
            rank: Process rank for distributed seed offset.
        """
        section = cfg.get("reproducibility", cfg) if hasattr(cfg, "get") else cfg
        seed = int(section.get("seed", 42))
        return cls(
            SeedConfig(
                seed=seed,
                cudnn_deterministic=bool(section.get("cudnn_deterministic", True)),
                cudnn_benchmark=bool(section.get("cudnn_benchmark", False)),
                allow_tf32=bool(section.get("allow_tf32", False)),
                rank=rank,
            )
        )


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    rank: int = 0,
) -> int:
    """
    One-shot seed initialization.

    Args:
        seed: Base random seed.
        deterministic: Enable cuDNN deterministic mode.
        benchmark: Enable cuDNN benchmark mode (faster, less reproducible).
        rank: Rank offset for distributed runs.

    Returns:
        Effective seed applied.
    """
    manager = ReproducibilityManager(
        SeedConfig(
            seed=seed,
            cudnn_deterministic=deterministic,
            cudnn_benchmark=benchmark,
            rank=rank,
        )
    )
    return manager.apply()
