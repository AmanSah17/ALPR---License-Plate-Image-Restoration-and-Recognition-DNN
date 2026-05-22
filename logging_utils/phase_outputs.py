"""
Per-phase output directory layout for modular pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

PathLike = Union[str, Path]


@dataclass
class PhaseOutputPaths:
    """
    Isolated folders for one pipeline phase under an experiment root.

    Example layout::

        experiments/{exp_id}/phases/optical_flow/
            flows/
            warped/
            overlays/
            error_maps/
            visualizations/
            metrics/
            logs/
    """

    root: Path
    flows: Path
    warped: Path
    overlays: Path
    error_maps: Path
    visualizations: Path
    metrics: Path
    logs: Path

    def as_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}

    def mkdirs(self) -> "PhaseOutputPaths":
        """Create all phase directories."""
        for path in (
            self.root,
            self.flows,
            self.warped,
            self.overlays,
            self.error_maps,
            self.visualizations,
            self.metrics,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


class PhaseOutputManager:
    """
    Factory for phase-specific output paths under an experiment directory.
    """

    def __init__(self, experiment_root: PathLike, phases_dirname: str = "phases") -> None:
        self.experiment_root = Path(experiment_root).resolve()
        self.phases_root = self.experiment_root / phases_dirname

    def get(self, phase_name: str) -> PhaseOutputPaths:
        """
        Resolve output paths for a named phase (e.g. ``optical_flow``).

        Args:
            phase_name: Phase identifier used as subdirectory name.

        Returns:
            ``PhaseOutputPaths`` with all subfolders (created on disk).
        """
        root = self.phases_root / phase_name
        paths = PhaseOutputPaths(
            root=root,
            flows=root / "flows",
            warped=root / "warped",
            overlays=root / "overlays",
            error_maps=root / "error_maps",
            visualizations=root / "visualizations",
            metrics=root / "metrics",
            logs=root / "logs",
        )
        return paths.mkdirs()
