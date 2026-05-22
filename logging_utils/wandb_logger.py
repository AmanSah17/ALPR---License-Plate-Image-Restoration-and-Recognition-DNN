"""
Optional Weights & Biases experiment logger.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]


class WandBLogger:
    """
    Optional W&B integration with graceful degradation when disabled or missing.

    Never raises on missing API keys when ``enabled=False``.
    """

    def __init__(
        self,
        project: str = "mf_lpr2",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = False,
        mode: str = "online",
        log_artifacts: bool = True,
        dir: Optional[PathLike] = None,
    ) -> None:
        """
        Args:
            project: W&B project name.
            entity: Team/user entity.
            name: Run name.
            config: Hyperparameter/config dict.
            tags: Run tags.
            enabled: Master switch.
            mode: ``online``, ``offline``, or ``disabled``.
            log_artifacts: Upload artifacts via W&B.
            dir: W&B sync directory.
        """
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.enabled = enabled and wandb is not None and mode != "disabled"
        self.mode = mode
        self.log_artifacts = log_artifacts
        self.dir = Path(dir) if dir else None
        self._run: Optional[Any] = None

        if enabled and wandb is None:
            logger.warning("wandb package not installed; WandBLogger disabled.")
            self.enabled = False

        if self.enabled:
            self._init_run()

    def _init_run(self) -> None:
        """Start W&B run."""
        if wandb is None:
            return
        kwargs: Dict[str, Any] = {
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "config": self.config,
            "tags": self.tags,
            "mode": self.mode,
        }
        if self.dir is not None:
            kwargs["dir"] = str(self.dir)
        self._run = wandb.init(**kwargs)
        logger.info("W&B run started: %s", self._run.url if self._run else "offline")

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics dict at step."""
        if not self.enabled or wandb is None:
            return
        wandb.log(data, step=step)

    def log_image(self, key: str, image_path: PathLike, caption: Optional[str] = None) -> None:
        """Log image from filesystem path."""
        if not self.enabled or wandb is None:
            return
        wandb.log({key: wandb.Image(str(image_path), caption=caption)})

    def log_artifact_file(
        self,
        file_path: PathLike,
        name: str,
        artifact_type: str = "output",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upload a file as a W&B artifact."""
        if not self.enabled or not self.log_artifacts or wandb is None:
            return
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(str(file_path))
        if self._run is not None:
            self._run.log_artifact(artifact)

    def finish(self) -> None:
        """End W&B run."""
        if self.enabled and wandb is not None and self._run is not None:
            wandb.finish()
            self._run = None

    def __enter__(self) -> "WandBLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        run_name: str,
        config_dict: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        log_dir: Optional[PathLike] = None,
    ) -> "WandBLogger":
        """Instantiate from YAML ``wandb`` section."""
        section = cfg.get("wandb", cfg) if hasattr(cfg, "get") else cfg
        return cls(
            project=str(section.get("project", "mf_lpr2")),
            entity=section.get("entity"),
            name=run_name,
            config=config_dict,
            tags=tags,
            enabled=bool(section.get("enabled", False)),
            mode=str(section.get("mode", "online")),
            log_artifacts=bool(section.get("log_artifacts", True)),
            dir=log_dir,
        )
