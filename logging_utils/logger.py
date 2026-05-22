"""
Structured logging factory with console, file, and JSON handlers.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            payload["extra"] = record.extra_data
        return json.dumps(payload, default=str)


@dataclass
class LoggingConfig:
    """Runtime logging configuration."""

    level: str = "INFO"
    console: bool = True
    file: bool = True
    json_file: bool = True
    rich_tracebacks: bool = True
    log_gpu_memory: bool = False
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


class LoggerFactory:
    """
    Create and configure hierarchical loggers for experiments.

    Supports rotating text logs and optional JSON logs for parsing in reports.
    """

    _configured_roots: Dict[str, bool] = {}

    def __init__(self, config: Optional[LoggingConfig] = None) -> None:
        self.config = config or LoggingConfig()

    @staticmethod
    def _level_from_string(level: str) -> int:
        mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return mapping.get(level.upper(), logging.INFO)

    def configure(
        self,
        log_dir: Optional[PathLike] = None,
        experiment_name: str = "mf_lpr2",
        root_name: str = "mf_lpr2",
    ) -> logging.Logger:
        """
        Configure the project root logger.

        Args:
            log_dir: Directory for log files; if ``None``, console only.
            experiment_name: Prefix for log filenames.
            root_name: Logger namespace (default ``mf_lpr2``).

        Returns:
            Configured root logger.
        """
        if self._configured_roots.get(root_name):
            return logging.getLogger(root_name)

        logger = logging.getLogger(root_name)
        logger.setLevel(self._level_from_string(self.config.level))
        logger.propagate = False
        logger.handlers.clear()

        text_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        if self.config.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logger.level)
            if self.config.rich_tracebacks:
                try:
                    from rich.logging import RichHandler

                    rich_handler = RichHandler(
                        rich_tracebacks=True,
                        show_time=True,
                        show_path=False,
                        markup=False,
                    )
                    rich_handler.setLevel(logger.level)
                    logger.addHandler(rich_handler)
                except ImportError:
                    console_handler.setFormatter(logging.Formatter(text_format, date_format))
                    logger.addHandler(console_handler)
            else:
                console_handler.setFormatter(logging.Formatter(text_format, date_format))
                logger.addHandler(console_handler)

        if self.config.file and log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path / f"{experiment_name}.log",
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(text_format, date_format))
            file_handler.setLevel(logger.level)
            logger.addHandler(file_handler)

            if self.config.json_file:
                json_handler = RotatingFileHandler(
                    log_path / f"{experiment_name}.json.log",
                    maxBytes=self.config.max_bytes,
                    backupCount=self.config.backup_count,
                    encoding="utf-8",
                )
                json_handler.setFormatter(JsonFormatter())
                json_handler.setLevel(logger.level)
                logger.addHandler(json_handler)

        self._configured_roots[root_name] = True
        logger.debug("Logger configured (level=%s, dir=%s)", self.config.level, log_dir)
        return logger

    @classmethod
    def from_omegaconf(cls, cfg: Any, log_dir: Optional[PathLike] = None) -> logging.Logger:
        """Build logger from config ``logging`` section."""
        section = cfg.get("logging", cfg) if hasattr(cfg, "get") else cfg
        factory = cls(
            LoggingConfig(
                level=str(section.get("level", "INFO")),
                console=bool(section.get("console", True)),
                file=bool(section.get("file", True)),
                json_file=bool(section.get("json_file", True)),
                rich_tracebacks=bool(section.get("rich_tracebacks", True)),
                log_gpu_memory=bool(section.get("log_gpu_memory", False)),
            )
        )
        return factory.configure(log_dir=log_dir)


def setup_logging(
    log_dir: PathLike,
    level: str = "INFO",
    experiment_name: str = "mf_lpr2",
) -> logging.Logger:
    """
    Convenience wrapper to configure default project logging.

    Args:
        log_dir: Log output directory.
        level: Log level string.
        experiment_name: Filename prefix.

    Returns:
        Root logger instance.
    """
    factory = LoggerFactory(LoggingConfig(level=level))
    return factory.configure(log_dir=log_dir, experiment_name=experiment_name)


def log_dict(logger: logging.Logger, message: str, data: Dict[str, Any], level: int = logging.INFO) -> None:
    """Log a message with structured extra payload (JSON file captures it)."""
    logger.log(level, message, extra={"extra_data": data})
