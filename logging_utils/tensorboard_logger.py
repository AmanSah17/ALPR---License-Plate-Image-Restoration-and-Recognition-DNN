"""
TensorBoard experiment logging wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc, assignment]


class TensorBoardLogger:
    """
    Thin wrapper around ``SummaryWriter`` with safe no-op fallback.

    Logs scalars, images, histograms, and hyperparameters for reproducibility.
    """

    def __init__(
        self,
        log_dir: PathLike,
        enabled: bool = True,
        flush_secs: int = 30,
        comment: str = "",
    ) -> None:
        """
        Args:
            log_dir: TensorBoard event directory.
            enabled: If False, all methods become no-ops.
            flush_secs: Writer flush interval.
            comment: Optional run comment suffix.
        """
        self.log_dir = Path(log_dir)
        self.enabled = enabled and SummaryWriter is not None
        self.flush_secs = flush_secs
        self.comment = comment
        self._writer: Optional[Any] = None

        if enabled and SummaryWriter is None:
            logger.warning("tensorboard not installed; TensorBoardLogger disabled.")
            self.enabled = False

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=flush_secs,
                comment=comment,
            )
            logger.info("TensorBoard logging enabled: %s", self.log_dir)

    @property
    def writer(self) -> Optional[Any]:
        """Underlying SummaryWriter or None."""
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar metric."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars under one main tag."""
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(
        self,
        tag: str,
        image_tensor,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """
        Log an image tensor.

        Args:
            tag: TensorBoard tag.
            image_tensor: ``torch.Tensor`` or compatible array.
            step: Global step.
            dataformats: Tensor layout (default CHW).
        """
        if self._writer is not None:
            self._writer.add_image(tag, image_tensor, step, dataformats=dataformats)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log weight or activation histogram."""
        if self._writer is not None:
            self._writer.add_histogram(tag, values, step)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Log hyperparameters with final metric dict."""
        if self._writer is not None:
            # TensorBoard requires flat hparams (str/int/float/bool)
            flat = {k: v for k, v in hparams.items() if isinstance(v, (str, int, float, bool))}
            self._writer.add_hparams(flat, metrics)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text snippet."""
        if self._writer is not None:
            self._writer.add_text(tag, text, step)

    def flush(self) -> None:
        """Flush pending events."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.debug("TensorBoard writer closed.")

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @classmethod
    def from_config(cls, cfg: Any, log_dir: PathLike) -> "TensorBoardLogger":
        """Create logger from YAML ``tensorboard`` section."""
        section = cfg.get("tensorboard", cfg) if hasattr(cfg, "get") else cfg
        return cls(
            log_dir=log_dir,
            enabled=bool(section.get("enabled", True)),
            flush_secs=int(section.get("flush_secs", 30)),
        )
