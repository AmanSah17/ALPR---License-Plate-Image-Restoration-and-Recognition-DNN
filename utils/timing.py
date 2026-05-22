"""
High-resolution timing utilities for inference benchmarking.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional


@dataclass
class TimerStats:
    """Aggregated timing statistics in milliseconds."""

    count: int
    total_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    fps: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": float(self.count),
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "fps": self.fps,
        }


class BenchmarkTimer:
    """
    Context-manager timer with warmup, rolling samples, and percentile stats.

    Uses ``time.perf_counter`` for sub-millisecond precision on Windows/Linux.
    """

    def __init__(self, name: str = "benchmark", warmup: int = 0) -> None:
        """
        Args:
            name: Logical name for reports.
            warmup: First N samples excluded from statistics.
        """
        self.name = name
        self.warmup = warmup
        self._samples: List[float] = []
        self._start: Optional[float] = None

    def start(self) -> None:
        """Begin a timing sample."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """
        End the current sample and record elapsed milliseconds.

        Returns:
            Elapsed time in milliseconds.

        Raises:
            RuntimeError: If ``start`` was not called.
        """
        if self._start is None:
            raise RuntimeError("BenchmarkTimer.stop() called without start().")
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._samples.append(elapsed_ms)
        self._start = None
        return elapsed_ms

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Context manager that records one sample."""
        self.start()
        try:
            yield
        finally:
            self.stop()

    @property
    def samples(self) -> List[float]:
        """All recorded samples (including warmup)."""
        return list(self._samples)

    def effective_samples(self) -> List[float]:
        """Samples after excluding warmup iterations."""
        if self.warmup <= 0:
            return self._samples
        return self._samples[self.warmup :]

    def stats(self) -> TimerStats:
        """
        Compute aggregate statistics from effective samples.

        Returns:
            ``TimerStats``; returns zeros if no samples yet.
        """
        data = self.effective_samples()
        if not data:
            return TimerStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        count = len(data)
        total = sum(data)
        mean = statistics.mean(data)
        std = statistics.pstdev(data) if count > 1 else 0.0
        sorted_data = sorted(data)
        p50 = sorted_data[int(0.50 * (count - 1))]
        p95 = sorted_data[int(0.95 * (count - 1))]
        fps = 1000.0 / mean if mean > 0 else 0.0
        return TimerStats(
            count=count,
            total_ms=total,
            mean_ms=mean,
            std_ms=std,
            min_ms=min(data),
            max_ms=max(data),
            p50_ms=p50,
            p95_ms=p95,
            fps=fps,
        )

    def reset(self) -> None:
        """Clear all samples."""
        self._samples.clear()
        self._start = None


@dataclass
class MultiStageTimer:
    """Named stage timers for pipeline profiling."""

    warmup: int = 0
    stages: Dict[str, BenchmarkTimer] = field(default_factory=dict)

    def stage(self, name: str) -> BenchmarkTimer:
        """Get or create a timer for a pipeline stage."""
        if name not in self.stages:
            self.stages[name] = BenchmarkTimer(name=name, warmup=self.warmup)
        return self.stages[name]

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return stats dict keyed by stage name."""
        return {name: timer.stats().to_dict() for name, timer in self.stages.items()}
