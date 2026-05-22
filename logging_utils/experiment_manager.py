"""
Timestamped experiment directory layout and unified tracking orchestration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from omegaconf import DictConfig, OmegaConf

from logging_utils.artifact_manager import ArtifactManager
from logging_utils.logger import LoggerFactory
from logging_utils.mlflow_logger import MLflowLogger
from logging_utils.tensorboard_logger import TensorBoardLogger
from utils.config_loader import ConfigManager
from utils.gpu_utils import GPUManager
from utils.seed_utils import ReproducibilityManager

PathLike = Union[str, Path]
logger = logging.getLogger(__name__)


@dataclass
class ExperimentPaths:
    """Standard experiment directory layout."""

    root: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    visualizations: Path
    predictions: Path
    reports: Path
    configs: Path
    mlruns: Path

    def as_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}


@dataclass
class ExperimentMetadata:
    """Serializable experiment metadata."""

    experiment_id: str
    name: str
    created_at: str
    project_root: str
    paths: Dict[str, str]
    seed: int
    mlflow_run_id: Optional[str] = None
    mlflow_experiment: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at,
            "project_root": self.project_root,
            "paths": self.paths,
            "seed": self.seed,
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_experiment": self.mlflow_experiment,
            "tags": self.tags,
            "notes": self.notes,
            "extra": self.extra,
        }


class ExperimentTracker:
    """
    Unified facade: filesystem layout + MLflow + optional TensorBoard.

    Ensures metrics logged in training/eval use consistent prefixes and steps.
    """

    METRIC_PREFIX_TRAIN = "train"
    METRIC_PREFIX_VAL = "val"
    METRIC_PREFIX_TEST = "test"
    METRIC_PREFIX_BENCH = "benchmark"

    def __init__(
        self,
        paths: ExperimentPaths,
        mlflow: MLflowLogger,
        tensorboard: Optional[TensorBoardLogger] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        metadata: Optional[ExperimentMetadata] = None,
    ) -> None:
        self.paths = paths
        self.mlflow = mlflow
        self.tensorboard = tensorboard
        self.artifacts = artifact_manager or ArtifactManager(paths.root)
        self.metadata = metadata

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        self.mlflow.log_params(params)

    def log_train_metrics(self, metrics: Mapping[str, Any], step: int) -> None:
        """
        Log training metrics at ``step`` with ``train/`` prefix.

        Also mirrors scalars to TensorBoard when enabled.
        """
        prefixed = {f"{self.METRIC_PREFIX_TRAIN}/{k}": v for k, v in _flatten_metrics(metrics).items()}
        self.mlflow.log_metrics(prefixed, step=step)
        self._mirror_tensorboard(prefixed, step)

    def log_val_metrics(self, metrics: Mapping[str, Any], step: int) -> None:
        """Log validation metrics with ``val/`` prefix."""
        prefixed = {f"{self.METRIC_PREFIX_VAL}/{k}": v for k, v in _flatten_metrics(metrics).items()}
        self.mlflow.log_metrics(prefixed, step=step)
        self._mirror_tensorboard(prefixed, step)

    def log_benchmark_metrics(self, metrics: Mapping[str, Any], step: int = 0) -> None:
        """Log inference benchmark metrics with ``benchmark/`` prefix."""
        prefixed = {
            f"{self.METRIC_PREFIX_BENCH}/{k}": v for k, v in _flatten_metrics(metrics).items()
        }
        self.mlflow.log_metrics(prefixed, step=step)
        self._mirror_tensorboard(prefixed, step)

    def _mirror_tensorboard(self, metrics: Dict[str, float], step: int) -> None:
        if self.tensorboard is None or not self.tensorboard.enabled:
            return
        for tag, value in metrics.items():
            self.tensorboard.log_scalar(tag, float(value), step)

    def save_metrics_json(self, metrics: Mapping[str, Any], filename: str, step: int) -> Path:
        """Persist metrics snapshot to ``paths.metrics`` and log as MLflow artifact."""
        out = self.paths.metrics / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": step, "metrics": dict(_flatten_metrics(metrics))}
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        self.mlflow.log_artifact(out, artifact_path="metrics")
        return out

    def close(self, status: str = "FINISHED") -> None:
        """Finalize all trackers."""
        if self.tensorboard is not None:
            self.tensorboard.close()
        self.mlflow.end(status=status)


class ExperimentManager:
    """
    Create timestamped experiment folders and wire MLflow + logging.
    """

    def __init__(self, project_root: PathLike, logging_cfg: Optional[DictConfig] = None) -> None:
        self.project_root = Path(project_root).resolve()
        if logging_cfg is None:
            logging_cfg = ConfigManager(self.project_root).load("logging", validate=True)
        self.cfg = logging_cfg
        self._exp_cfg = self.cfg.experiment
        self._dir_cfg = self.cfg.directories

    def _build_experiment_id(self, name: str) -> str:
        fmt = str(self._exp_cfg.get("timestamp_format", "%Y%m%d_%H%M%S"))
        timestamp = datetime.now().strftime(fmt)
        if bool(self._exp_cfg.get("auto_name", True)):
            return f"{timestamp}_{name}"
        return name

    def create(
        self,
        name: str = "experiment",
        tags: Optional[List[str]] = None,
        notes: str = "",
        config_snapshot: Optional[DictConfig] = None,
        seed: int = 42,
        tracking_uri_override: Optional[str] = None,
    ) -> ExperimentTracker:
        """
        Initialize a new experiment with directories and trackers.

        Args:
            name: Short experiment name.
            tags: Tags for MLflow and metadata.
            config_snapshot: Config to save under ``configs/``.
            seed: Reproducibility seed applied immediately.
            tracking_uri_override: Optional MLflow URI override.

        Returns:
            ``ExperimentTracker`` ready for metric logging.
        """
        exp_id = self._build_experiment_id(name)
        base = self.project_root / str(self._exp_cfg.get("base_dir", "outputs"))
        sub = str(self._exp_cfg.get("experiments_subdir", "experiments"))
        root = (base / sub / exp_id).resolve()

        paths = ExperimentPaths(
            root=root,
            checkpoints=root / self._dir_cfg.checkpoints,
            logs=root / self._dir_cfg.logs,
            metrics=root / self._dir_cfg.metrics,
            visualizations=root / self._dir_cfg.visualizations,
            predictions=root / self._dir_cfg.predictions,
            reports=root / self._dir_cfg.reports,
            configs=root / self._dir_cfg.configs,
            mlruns=base / "mlruns",
        )
        for p in (
            paths.checkpoints,
            paths.logs,
            paths.metrics,
            paths.visualizations,
            paths.predictions,
            paths.reports,
            paths.configs,
        ):
            p.mkdir(parents=True, exist_ok=True)

        # Reproducibility
        repro = ReproducibilityManager.from_omegaconf({"reproducibility": {"seed": seed}})
        repro.apply()

        # File logging
        LoggerFactory.from_omegaconf(self.cfg, log_dir=paths.logs)

        # GPU status in logs
        if bool(self.cfg.logging.get("log_gpu_memory", False)):
            GPUManager.from_config(self.cfg.gpu).log_status()

        # Config snapshot
        if config_snapshot is not None and bool(self._exp_cfg.get("save_config_snapshot", True)):
            ConfigManager(self.project_root).snapshot_to_experiment(
                config_snapshot, paths.configs
            )

        # MLflow — tracking URI defaults to experiment mlruns folder
        mlflow_section = self.cfg.mlflow
        default_uri = str(paths.mlruns.resolve().as_uri()) if paths.mlruns else None
        if tracking_uri_override is None and mlflow_section.get("tracking_uri") in (None, "auto"):
            tracking_uri = f"file:///{paths.mlruns.resolve().as_posix()}"
        else:
            tracking_uri = tracking_uri_override or mlflow_section.get("tracking_uri")

        tag_dict = {str(t): "true" for t in (tags or [])}
        mlflow_logger = MLflowLogger.from_config(
            mlflow_section,
            run_name=exp_id,
            tags=tag_dict,
            tracking_uri_override=tracking_uri,
        )
        mlflow_logger.start()

        # Log full config as params (flattened)
        if config_snapshot is not None:
            resolved = OmegaConf.to_container(config_snapshot, resolve=True)
            mlflow_logger.log_params(resolved)  # type: ignore[arg-type]

        tensorboard = None
        if bool(self.cfg.tensorboard.get("enabled", False)):
            tensorboard = TensorBoardLogger.from_config(
                self.cfg, log_dir=paths.logs / "tensorboard"
            )

        metadata = ExperimentMetadata(
            experiment_id=exp_id,
            name=name,
            created_at=datetime.now().isoformat(),
            project_root=str(self.project_root),
            paths=paths.as_dict(),
            seed=seed,
            mlflow_run_id=mlflow_logger.run_id,
            mlflow_experiment=str(mlflow_section.get("experiment_name", "mf_lpr2")),
            tags=tags or [],
            notes=notes,
        )

        if bool(self._exp_cfg.get("save_metadata", True)):
            meta_path = root / str(self.cfg.artifacts.get("metadata_filename", "experiment_metadata.json"))
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata.to_dict(), handle, indent=2)

        artifact_mgr = ArtifactManager(
            root,
            versioning=bool(self.cfg.artifacts.get("versioning", True)),
        )

        logger.info("Experiment created: %s", root)
        return ExperimentTracker(
            paths=paths,
            mlflow=mlflow_logger,
            tensorboard=tensorboard,
            artifact_manager=artifact_mgr,
            metadata=metadata,
        )


def _flatten_metrics(metrics: Mapping[str, Any], parent: str = "") -> Dict[str, Any]:
    """Flatten nested metric dicts to slash-separated keys."""
    out: Dict[str, Any] = {}
    for key, val in metrics.items():
        full_key = f"{parent}/{key}" if parent else str(key)
        if isinstance(val, Mapping):
            out.update(_flatten_metrics(val, full_key))
        else:
            out[full_key] = val
    return out
