"""
YAML configuration management with merging, validation, and snapshots.

Supports OmegaConf-style defaults lists and deep merging for experiment
reproducibility.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


class ConfigLoader:
    """
    Load and merge YAML configuration files.

    Attributes:
        config_dir: Root directory containing YAML config files.
    """

    def __init__(self, config_dir: PathLike) -> None:
        """
        Initialize the loader.

        Args:
            config_dir: Directory path that stores ``*.yaml`` configs.
        """
        self.config_dir = Path(config_dir).resolve()
        if not self.config_dir.is_dir():
            raise ConfigError(f"Config directory does not exist: {self.config_dir}")

    def _resolve_path(self, name: PathLike) -> Path:
        """Resolve a config name or path relative to ``config_dir``."""
        path = Path(name)
        if path.suffix == "":
            path = path.with_suffix(".yaml")
        if not path.is_absolute():
            path = self.config_dir / path
        return path.resolve()

    def load_yaml(self, name: PathLike) -> Dict[str, Any]:
        """
        Load a single YAML file into a plain dictionary.

        Args:
            name: Config filename (e.g. ``train.yaml``) or absolute path.

        Returns:
            Parsed configuration dictionary.

        Raises:
            ConfigError: If the file is missing or cannot be parsed.
        """
        path = self._resolve_path(name)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML: {path}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"Root YAML node must be a mapping: {path}")
        return data

    def _load_defaults(self, defaults: Sequence[Any], visited: List[str]) -> Dict[str, Any]:
        """
        Recursively load ``defaults`` entries (OmegaConf convention).

        Args:
            defaults: List of default config references from YAML.
            visited: Chain of already-loaded files for cycle detection.

        Returns:
            Merged dictionary from all default files.
        """
        merged: Dict[str, Any] = {}
        for entry in defaults:
            if isinstance(entry, str):
                ref = entry
                overrides: Dict[str, Any] = {}
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ConfigError(f"Invalid defaults entry: {entry}")
                ref, overrides = next(iter(entry.items()))
            else:
                raise ConfigError(f"Unsupported defaults entry type: {type(entry)}")

            if ref in visited:
                raise ConfigError(f"Circular config reference detected: {ref}")

            base = self.load_yaml(ref)
            if "defaults" in base:
                nested = self._load_defaults(base.pop("defaults"), visited + [ref])
                base = OmegaConf.to_container(
                    OmegaConf.merge(OmegaConf.create(nested), OmegaConf.create(base)),
                    resolve=True,
                )
            merged = OmegaConf.to_container(
                OmegaConf.merge(OmegaConf.create(merged), OmegaConf.create(base)),
                resolve=True,
            )
            if overrides:
                merged = OmegaConf.to_container(
                    OmegaConf.merge(
                        OmegaConf.create(merged),
                        OmegaConf.create(overrides),
                    ),
                    resolve=True,
                )
        return merged  # type: ignore[return-value]

    def load(self, name: PathLike, overrides: Optional[Mapping[str, Any]] = None) -> DictConfig:
        """
        Load a config with optional ``defaults`` chain and runtime overrides.

        Args:
            name: Primary config file.
            overrides: Dotlist-style or nested dict overrides applied last.

        Returns:
            OmegaConf ``DictConfig`` (supports dot access and ``OmegaConf.merge``).
        """
        raw = self.load_yaml(name)
        defaults = raw.pop("defaults", None)
        base: Dict[str, Any] = {}
        if defaults is not None:
            base = self._load_defaults(defaults, visited=[str(name)])
        cfg = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(raw))
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(overrides)))
        return cfg

    def save_snapshot(
        self,
        cfg: Union[DictConfig, Mapping[str, Any]],
        destination: PathLike,
        filename: str = "config_snapshot.yaml",
    ) -> Path:
        """
        Persist a resolved config snapshot for experiment reproducibility.

        Args:
            cfg: Configuration to save.
            destination: Directory where the snapshot is written.
            filename: Output filename.

        Returns:
            Path to the written snapshot file.
        """
        dest_dir = Path(destination)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / filename
        resolved = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, default_flow_style=False, sort_keys=False)
        logger.info("Saved config snapshot: %s", out_path)
        return out_path

    def to_json(self, cfg: Union[DictConfig, Mapping[str, Any]]) -> str:
        """Serialize config to a JSON string."""
        resolved = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
        return json.dumps(resolved, indent=2, default=str)


class ConfigManager:
    """
    High-level configuration facade for training and inference entry points.

    Combines loading, validation hooks, and immutable snapshots for experiments.
    """

    REQUIRED_KEYS: Dict[str, List[str]] = {
        "train": ["trainer", "optimizer", "reproducibility"],
        "inference": ["inference", "benchmark"],
        "dataset": ["dataset", "loading"],
        "model": ["optical_flow", "restoration"],
        "logging": ["logging", "experiment", "directories"],
    }

    def __init__(self, project_root: PathLike, config_dir: Optional[PathLike] = None) -> None:
        """
        Args:
            project_root: Repository root (parent of ``configs/``).
            config_dir: Optional override for config directory.
        """
        self.project_root = Path(project_root).resolve()
        cfg_dir = config_dir or self.project_root / "configs"
        self.loader = ConfigLoader(cfg_dir)

    def load(
        self,
        profile: str,
        overrides: Optional[Mapping[str, Any]] = None,
        validate: bool = True,
    ) -> DictConfig:
        """
        Load a named configuration profile.

        Args:
            profile: One of ``train``, ``inference``, ``model``, ``dataset``, ``logging``.
            overrides: Runtime overrides.
            validate: Whether to run required-key validation.

        Returns:
            Resolved ``DictConfig``.
        """
        filename = f"{profile}.yaml"
        cfg = self.loader.load(filename, overrides=overrides)
        OmegaConf.set_struct(cfg, False)
        if validate and profile in self.REQUIRED_KEYS:
            self._validate_required(cfg, self.REQUIRED_KEYS[profile], profile)
        return cfg

    def _validate_required(self, cfg: DictConfig, keys: List[str], profile: str) -> None:
        """Ensure top-level required keys exist."""
        missing = [key for key in keys if OmegaConf.select(cfg, key) is None]
        if missing:
            raise ConfigError(
                f"Profile '{profile}' is missing required keys: {missing}"
            )

    def merge_profiles(self, *profiles: str, overrides: Optional[Mapping[str, Any]] = None) -> DictConfig:
        """
        Load and deep-merge multiple profiles (left-to-right precedence).

        Args:
            *profiles: Config profile names without ``.yaml`` extension.
            overrides: Final override mapping.

        Returns:
            Merged ``DictConfig``.
        """
        merged: Optional[DictConfig] = None
        for name in profiles:
            part = self.loader.load(f"{name}.yaml", overrides=None)
            merged = part if merged is None else OmegaConf.merge(merged, part)
        if overrides:
            merged = OmegaConf.merge(merged, OmegaConf.create(dict(overrides)))  # type: ignore[arg-type]
        return merged  # type: ignore[return-value]

    def snapshot_to_experiment(
        self,
        cfg: DictConfig,
        experiment_config_dir: PathLike,
    ) -> Path:
        """Save config snapshot into an experiment's ``configs`` subdirectory."""
        return self.loader.save_snapshot(cfg, experiment_config_dir)


def load_config(
    project_root: PathLike,
    profile: str = "train",
    overrides: Optional[Mapping[str, Any]] = None,
) -> DictConfig:
    """
    Convenience function to load a configuration profile.

    Args:
        project_root: Repository root path.
        profile: Config profile name.
        overrides: Optional runtime overrides.

    Returns:
        Loaded ``DictConfig``.
    """
    manager = ConfigManager(project_root)
    return manager.load(profile, overrides=overrides)


def config_from_dict(data: Mapping[str, Any]) -> DictConfig:
    """Create a ``DictConfig`` from an in-memory mapping."""
    return OmegaConf.create(copy.deepcopy(dict(data)))
