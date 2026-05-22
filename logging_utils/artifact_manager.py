"""
Versioned artifact storage linked to experiment directories and MLflow.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


@dataclass
class ArtifactRecord:
    """Metadata for a stored artifact."""

    name: str
    category: str
    path: str
    version: int
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "path": self.path,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class ArtifactManager:
    """
    Manage checkpoints, visualizations, predictions, and metric exports.

    Maintains a manifest JSON for reproducibility and MLflow artifact sync.
    """

    MANIFEST_NAME = "artifact_manifest.json"
    CATEGORIES = ("checkpoints", "visualizations", "predictions", "metrics", "reports")

    def __init__(
        self,
        experiment_root: PathLike,
        versioning: bool = True,
    ) -> None:
        """
        Args:
            experiment_root: Root experiment directory.
            versioning: Append version suffix when name collides.
        """
        self.root = Path(experiment_root).resolve()
        self.versioning = versioning
        self._manifest_path = self.root / self.MANIFEST_NAME
        self._records: List[ArtifactRecord] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self._manifest_path.exists():
            with self._manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._records = [
                ArtifactRecord(**item) for item in data.get("artifacts", [])
            ]

    def _save_manifest(self) -> None:
        payload = {
            "experiment_root": str(self.root),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": [r.to_dict() for r in self._records],
        }
        with self._manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _category_dir(self, category: str) -> Path:
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Use one of {self.CATEGORIES}")
        path = self.root / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _next_version(self, name: str, category: str) -> int:
        versions = [
            r.version for r in self._records if r.name == name and r.category == category
        ]
        return max(versions, default=0) + 1

    def register(
        self,
        source_path: PathLike,
        name: str,
        category: str,
        copy_file: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        """
        Register (and optionally copy) an artifact into the experiment tree.

        Args:
            source_path: Existing file to register.
            name: Logical artifact name.
            category: One of ``CATEGORIES``.
            copy_file: If True, copy into category subdirectory.
            metadata: Optional JSON-serializable metadata.

        Returns:
            ``ArtifactRecord`` for the stored artifact.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source artifact not found: {src}")

        version = self._next_version(name, category) if self.versioning else 1
        dest_dir = self._category_dir(category)
        suffix = src.suffix
        dest_name = f"{name}_v{version}{suffix}" if self.versioning and version > 1 else f"{name}{suffix}"
        dest = dest_dir / dest_name

        if copy_file:
            shutil.copy2(src, dest)
        else:
            dest = src.resolve()

        record = ArtifactRecord(
            name=name,
            category=category,
            path=str(dest),
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self._records.append(record)
        self._save_manifest()
        logger.info("Registered artifact [%s/%s] -> %s", category, name, dest)
        return record

    def list_artifacts(self, category: Optional[str] = None) -> List[ArtifactRecord]:
        """List registered artifacts, optionally filtered by category."""
        if category is None:
            return list(self._records)
        return [r for r in self._records if r.category == category]

    def latest(self, name: str, category: str) -> Optional[ArtifactRecord]:
        """Return highest-version artifact for name/category."""
        matches = [r for r in self._records if r.name == name and r.category == category]
        if not matches:
            return None
        return max(matches, key=lambda r: r.version)

    def export_manifest(self, path: Optional[PathLike] = None) -> Path:
        """Write manifest copy to reports directory."""
        out = Path(path) if path else self.root / "reports" / "artifact_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._manifest_path, out)
        return out
