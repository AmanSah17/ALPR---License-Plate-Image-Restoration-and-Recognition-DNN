"""
Memory and compute profiling helpers for benchmarking reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.gpu_utils import GPUManager, GPUMemoryStats
from utils.timing import BenchmarkTimer, MultiStageTimer, TimerStats

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class ProfileReport:
    """Combined timing and memory profile for one run."""

    name: str
    timing: Dict[str, float]
    memory: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timing": self.timing,
            "memory": self.memory,
            "metadata": self.metadata,
        }


class RuntimeProfiler:
    """
    Orchestrates stage timers, GPU memory snapshots, and optional FLOP counting.

    Intended for inference benchmark scripts (Phase 3+).
    """

    def __init__(
        self,
        name: str = "profile",
        gpu_manager: Optional[GPUManager] = None,
        warmup: int = 5,
    ) -> None:
        self.name = name
        self.gpu = gpu_manager or GPUManager()
        self.timer = MultiStageTimer(warmup=warmup)
        self._memory_before: Optional[GPUMemoryStats] = None
        self._memory_after: Optional[GPUMemoryStats] = None

    def begin_run(self) -> None:
        """Reset peak memory and capture baseline."""
        self.gpu.reset_peak_memory()
        self.gpu.empty_cache()
        self._memory_before = self.gpu.memory_stats()

    def end_run(self) -> ProfileReport:
        """Finalize run and build report."""
        self.gpu.maybe_empty_cache()
        self._memory_after = self.gpu.memory_stats()
        timing_summary = self.timer.summary()
        # Use total stage if present, else sum means
        if "total" in timing_summary:
            primary = timing_summary["total"]
        else:
            primary = {"mean_ms": sum(s.get("mean_ms", 0) for s in timing_summary.values())}
        memory = self._build_memory_dict()
        return ProfileReport(
            name=self.name,
            timing=primary,
            memory=memory,
            metadata={"stages": timing_summary},
        )

    def _build_memory_dict(self) -> Dict[str, float]:
        after = self._memory_after or self.gpu.memory_stats()
        before = self._memory_before or after
        return {
            "allocated_mb_before": before.allocated_mb,
            "allocated_mb_after": after.allocated_mb,
            "delta_allocated_mb": after.allocated_mb - before.allocated_mb,
            "peak_allocated_mb": after.max_allocated_mb,
            "reserved_mb_after": after.reserved_mb,
        }

    def save_json(self, report: ProfileReport, path: Path) -> Path:
        """Write profile report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        logger.info("Saved profile report: %s", path)
        return path


def count_parameters(model: "nn.Module", trainable_only: bool = False) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch module.
        trainable_only: If True, count only ``requires_grad`` parameters.

    Returns:
        Parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_flops(
    model: "nn.Module",
    input_tensor: "torch.Tensor",
) -> Optional[float]:
    """
    Estimate FLOPs using ``thop`` if installed.

    Args:
        model: Module to profile.
        input_tensor: Example input on correct device/dtype.

    Returns:
        FLOPs as float, or ``None`` if ``thop`` unavailable.
    """
    try:
        from thop import profile  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("thop not installed; FLOP estimation skipped.")
        return None
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    # MACs ~ 2 * FLOPs convention; thop returns MACs
    return float(macs) * 2.0


def run_timed_iterations(
    fn,
    timer: BenchmarkTimer,
    iterations: int,
    warmup: int,
) -> TimerStats:
    """
    Execute a callable for benchmark iterations.

    Args:
        fn: Zero-argument callable to time.
        timer: ``BenchmarkTimer`` instance.
        iterations: Total iterations including warmup.
        warmup: Warmup count (should match timer.warmup).

    Returns:
        ``TimerStats`` after all iterations.
    """
    timer.warmup = warmup
    for _ in range(iterations):
        with timer.measure():
            fn()
    return timer.stats()
