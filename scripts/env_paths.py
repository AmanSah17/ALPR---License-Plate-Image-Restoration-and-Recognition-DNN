"""
Canonical paths for the CUDA + Numba enabled gemma4 virtual environment.

All project scripts should resolve Python via ``get_python_executable()`` rather
than relying on the system default interpreter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# User-specified environment (Scripts, not Scriots)
GEMMA4_VENV_ROOT = Path(r"D:\gemma4\gemma4")
GEMMA4_ACTIVATE_BAT = GEMMA4_VENV_ROOT / "Scripts" / "activate.bat"
GEMMA4_ACTIVATE_PS1 = GEMMA4_VENV_ROOT / "Scripts" / "Activate.ps1"
GEMMA4_PYTHON = GEMMA4_VENV_ROOT / "Scripts" / "python.exe"


def get_venv_root() -> Path:
    """Return gemma4 venv root; override with ``MF_LPR_VENV`` env var."""
    override = os.environ.get("MF_LPR_VENV")
    if override:
        return Path(override).resolve()
    return GEMMA4_VENV_ROOT.resolve()


def get_python_executable() -> Path:
    """
    Resolve the project Python interpreter.

    Priority:
        1. ``MF_LPR_PYTHON`` environment variable
        2. gemma4 venv ``Scripts/python.exe``
        3. Current ``sys.executable``
    """
    env_python = os.environ.get("MF_LPR_PYTHON")
    if env_python:
        return Path(env_python).resolve()
    root = get_venv_root()
    candidate = root / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    return Path(sys.executable).resolve()


def ensure_venv_on_path() -> Path:
    """
    If not already running under gemma4, print a warning (do not hard-exit).

    Returns:
        Resolved Python executable path.
    """
    expected = get_python_executable()
    current = Path(sys.executable).resolve()
    if current != expected.resolve():
        import warnings

        warnings.warn(
            f"Not using gemma4 interpreter. Expected {expected}, got {current}. "
            f"Activate with: {GEMMA4_ACTIVATE_PS1}",
            stacklevel=2,
        )
    return expected


def activation_command(shell: str = "powershell") -> str:
    """
    Return the activation command string for documentation/CLI hints.

    Args:
        shell: ``powershell`` or ``cmd``.
    """
    if shell.lower() in ("cmd", "batch", "bat"):
        return f'call "{GEMMA4_ACTIVATE_BAT}"'
    return f'& "{GEMMA4_ACTIVATE_PS1}"'
