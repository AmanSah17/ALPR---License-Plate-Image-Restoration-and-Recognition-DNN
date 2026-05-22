from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


def _experiment_dirs_for_model(experiments_dir: Path, model_name: str) -> Iterable[Path]:
    patterns = (
        f"*_train_{model_name}*",
        f"*_joint_p7_{model_name}*",
        f"*{model_name}*",
    )
    seen = set()
    for pattern in patterns:
        for path in sorted(experiments_dir.glob(pattern)):
            if path.is_dir() and path not in seen:
                seen.add(path)
                yield path


def _metric_from_checkpoint_name(path: Path, metric_name: str) -> Optional[float]:
    patterns = {
        "psnr": r"(?:val_psnr=|psnr=)([-+]?\d+(?:\.\d+)?)",
        "cer": r"(?:val_cer=|cer=)([-+]?\d+(?:\.\d+)?)",
    }
    pattern = patterns.get(metric_name.lower())
    if not pattern:
        return None
    match = re.search(pattern, path.name)
    return float(match.group(1)) if match else None


def resolve_best_checkpoint(
    model_name: str,
    experiments_dir: str | Path = "outputs/experiments",
    prefer_metric: str = "psnr",
) -> Path:
    """
    Resolve the best checkpoint for a model across restoration and joint runs.

    For ``prefer_metric='psnr'`` the highest PSNR-named checkpoint is returned.
    For ``prefer_metric='cer'`` the lowest CER-named checkpoint is returned.
    Falls back to the newest ``last.ckpt`` when metric-tagged checkpoints do not exist.
    """

    base = Path(experiments_dir)
    if not base.exists():
        raise FileNotFoundError(f"Experiments directory not found: {base}")

    ckpt_candidates = []
    last_candidates = []
    for exp_dir in _experiment_dirs_for_model(base, model_name):
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        ckpt_candidates.extend(sorted(ckpt_dir.glob("best*.ckpt")))
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            last_candidates.append(last_ckpt)

    if ckpt_candidates:
        metric = prefer_metric.lower()
        scored = []
        for ckpt in ckpt_candidates:
            score = _metric_from_checkpoint_name(ckpt, metric)
            if score is not None:
                scored.append((score, ckpt))
        if scored:
            if metric == "cer":
                return min(scored, key=lambda item: item[0])[1]
            return max(scored, key=lambda item: item[0])[1]
        return sorted(ckpt_candidates)[-1]

    if last_candidates:
        return sorted(last_candidates)[-1]

    raise FileNotFoundError(f"No checkpoint found for model '{model_name}' in {base}")
