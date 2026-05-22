"""
MLflow experiment tracking with type-safe, step-accurate metric logging.

Primary tracker for the MF-LPR2 framework. TensorBoard remains optional for
local live curves; all canonical metrics are logged to MLflow.
"""

from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment,misc]


def _coerce_float(value: Any) -> float:
    """
    Convert a metric value to a finite float for MLflow.

    Args:
        value: Scalar metric (int, float, numpy scalar, tensor item).

    Returns:
        Python float.

    Raises:
        TypeError: If value cannot be converted.
        ValueError: If value is NaN or infinite.
    """
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, bool):
        raise TypeError("Boolean metrics must not be logged as floats; use log_param.")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"Metric must be numeric, got {type(value).__name__}: {value!r}")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf check
        raise ValueError(f"Metric must be finite, got {value!r}")
    return out


def _normalize_metrics(
    metrics: Mapping[str, Any],
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    """
    Validate and flatten a metrics dictionary for MLflow.

    Nested keys use slash notation: ``{"train": {"loss": 0.1}}`` ->
    ``train/loss``.

    Args:
        metrics: Raw metric mapping (possibly nested).
        prefix: Optional key prefix applied to all keys.

    Returns:
        Flat ``{name: float}`` dict with finite floats only.
    """
    flat: Dict[str, float] = {}

    def _walk(data: Mapping[str, Any], parent: str) -> None:
        for key, val in data.items():
            name = f"{parent}/{key}" if parent else str(key)
            if isinstance(val, Mapping):
                _walk(val, name)
            else:
                flat[name] = _coerce_float(val)

    _walk(metrics, prefix or "")
    return flat


class MLflowLogger:
    """
    MLflow run lifecycle manager with accurate step-based metric logging.

    Usage::

        with MLflowLogger.from_config(cfg, run_name="exp_001") as tracker:
            tracker.log_params({"lr": 1e-4})
            tracker.log_metrics({"train/loss": 0.5}, step=0)
    """

    def __init__(
        self,
        experiment_name: str = "mf_lpr2",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        enabled: bool = True,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
        parent_run_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            experiment_name: MLflow experiment name (created if missing).
            run_name: Human-readable run name.
            tracking_uri: MLflow tracking URI; defaults to local ``mlruns``.
            artifact_location: Optional artifact root override.
            enabled: Master switch; when False, all methods no-op.
            tags: Run tags (string values only).
            nested: Whether this is a nested child run.
            parent_run_id: Parent run ID for nested runs.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.enabled = enabled and mlflow is not None
        self.tags = tags or {}
        self.nested = nested
        self.parent_run_id = parent_run_id
        self._run_id: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._global_step: int = 0
        self._last_step_per_prefix: Dict[str, int] = {}

        if enabled and mlflow is None:
            logger.warning("mlflow not installed; MLflowLogger disabled.")
            self.enabled = False

    @property
    def run_id(self) -> Optional[str]:
        """Active MLflow run ID."""
        return self._run_id

    @property
    def is_active(self) -> bool:
        """True if a run is currently active."""
        return self.enabled and self._run_id is not None

    def start(self) -> "MLflowLogger":
        """
        Start or resume an MLflow run.

        Returns:
            Self for chaining.
        """
        if not self.enabled or mlflow is None:
            return self

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            exp_id = client.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location,
            )
        else:
            exp_id = experiment.experiment_id
        self._experiment_id = exp_id

        run = mlflow.start_run(
            experiment_id=exp_id,
            run_name=self.run_name,
            nested=self.nested,
            parent_run_id=self.parent_run_id,
        )
        self._run_id = run.info.run_id

        if self.tags:
            mlflow.set_tags({k: str(v) for k, v in self.tags.items()})

        logger.info(
            "MLflow run started: experiment=%s run_id=%s name=%s",
            self.experiment_name,
            self._run_id,
            self.run_name,
        )
        return self

    def end(self, status: str = "FINISHED") -> None:
        """
        End the active MLflow run.

        Args:
            status: MLflow run status string (``FINISHED``, ``FAILED``, ``KILLED``).
        """
        if not self.enabled or mlflow is None:
            return
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run(status=status)
        finally:
            self._run_id = None
            logger.debug("MLflow run ended (status=%s).", status)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """
        Log hyperparameters / config (one-time, not step-indexed).

        Non-scalar values are JSON-serialized as strings.

        Args:
            params: Flat or nested parameter dict.
        """
        if not self.is_active or mlflow is None:
            return
        flat: Dict[str, str] = {}
        for key, val in _flatten_dict(params).items():
            if isinstance(val, (list, tuple)):
                flat[key] = ",".join(str(x) for x in val)
            else:
                flat[key] = str(val)
        # MLflow batch limit — chunk if needed
        for k, v in flat.items():
            mlflow.log_param(k, v[:250] if len(v) > 250 else v)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """
        Log scalar metrics at an explicit global step.

        MLflow requires monotonically increasing steps per metric name when
        using the same step argument across calls. This method enforces integer
        steps and rejects invalid values.

        Args:
            metrics: Metric name -> value mapping (supports nesting).
            step: Global step (epoch, iteration, or frame index). Required
                for training loops; defaults to internal counter if None.
            prefix: Optional prefix prepended to all keys (e.g. ``val``).
        """
        if not self.is_active or mlflow is None:
            return

        resolved_step = self._resolve_step(step, prefix)
        flat = _normalize_metrics(metrics, prefix=prefix)

        for name, value in flat.items():
            mlflow.log_metric(name, value, step=resolved_step)

        logger.debug(
            "MLflow metrics step=%d keys=%s",
            resolved_step,
            list(flat.keys()),
        )

    def log_metric(
        self,
        key: str,
        value: Any,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Log a single metric (convenience wrapper)."""
        name = f"{prefix}/{key}" if prefix else key
        self.log_metrics({name: value}, step=step)

    def set_step(self, step: int) -> None:
        """Set the default global step for subsequent metric logs."""
        if step < 0:
            raise ValueError(f"Step must be non-negative, got {step}")
        self._global_step = step

    def increment_step(self, delta: int = 1) -> int:
        """Advance and return the internal global step."""
        self._global_step += delta
        return self._global_step

    def _resolve_step(self, step: Optional[int], prefix: Optional[str]) -> int:
        if step is None:
            step = self._global_step
        if step < 0:
            raise ValueError(f"Step must be non-negative, got {step}")
        key = prefix or "__global__"
        last = self._last_step_per_prefix.get(key, -1)
        if step < last:
            logger.warning(
                "Non-monotonic step for prefix '%s': %d < %d (MLflow may overwrite)",
                key,
                step,
                last,
            )
        self._last_step_per_prefix[key] = step
        return int(step)

    def log_artifact(self, local_path: PathLike, artifact_path: Optional[str] = None) -> None:
        """
        Log a local file or directory to the run artifact store.

        Args:
            local_path: File or directory on disk.
            artifact_path: Subdirectory inside the run artifact root.
        """
        if not self.is_active or mlflow is None:
            return
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, local_dir: PathLike, artifact_path: Optional[str] = None) -> None:
        """Log all files under a directory."""
        if not self.is_active or mlflow is None:
            return
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_dict(self, dictionary: Mapping[str, Any], artifact_file: str) -> None:
        """Log a JSON-serializable dict as a run artifact file."""
        if not self.is_active or mlflow is None:
            return
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(dict(dictionary), tmp, indent=2, default=str)
            tmp_path = tmp.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path=artifact_file)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def log_image(self, key: str, image_path: PathLike, step: Optional[int] = None) -> None:
        """
        Log an image artifact; also records image path as metric tag at step.

        For MLflow 2.x+, uses ``log_image`` when available.
        """
        if not self.is_active or mlflow is None:
            return
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            import mlflow.images  # type: ignore[attr-defined]

            mlflow.log_image(path, key=key, step=step)
        except (ImportError, AttributeError, TypeError):
            mlflow.log_artifact(str(path), artifact_path=f"images/{key}")

    def get_run_url(self) -> Optional[str]:
        """Return UI URL for the active run if tracking server supports it."""
        if not self.is_active or mlflow is None:
            return None
        try:
            return mlflow.get_run(self._run_id).info.run_uri  # type: ignore[arg-type]
        except Exception:
            return None

    def __enter__(self) -> "MLflowLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end(status=status)

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        tracking_uri_override: Optional[str] = None,
    ) -> "MLflowLogger":
        """
        Build logger from YAML ``mlflow`` section in ``logging.yaml``.

        Args:
            cfg: Full logging config or ``mlflow`` subsection.
            run_name: Run display name.
            tags: Additional run tags.
            tracking_uri_override: CLI override for tracking URI.
        """
        section = cfg.get("mlflow", cfg) if hasattr(cfg, "get") else cfg
        merged_tags = {str(k): str(v) for k, v in section.get("tags", {}).items()}
        if tags:
            merged_tags.update({k: str(v) for k, v in tags.items()})
        uri = tracking_uri_override or section.get("tracking_uri")
        return cls(
            experiment_name=str(section.get("experiment_name", "mf_lpr2")),
            run_name=run_name,
            tracking_uri=uri,
            artifact_location=section.get("artifact_location"),
            enabled=bool(section.get("enabled", True)),
            tags=merged_tags,
            nested=bool(section.get("nested", False)),
        )


def _flatten_dict(
    data: Mapping[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten nested dict for MLflow params."""
    items: Dict[str, Any] = {}
    for key, val in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(val, Mapping):
            items.update(_flatten_dict(val, new_key, sep=sep))
        else:
            items[new_key] = val
    return items
