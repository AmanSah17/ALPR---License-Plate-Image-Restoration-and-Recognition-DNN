@echo off
REM Activate gemma4 CUDA + Numba environment for MF-LPR2 development.
REM Usage (cmd):
REM   call scripts\activate_env.bat
REM   python scripts\verify_phase1.py

if defined MF_LPR_VENV (
    set "VENV_ROOT=%MF_LPR_VENV%"
) else (
    set "VENV_ROOT=D:\gemma4\gemma4"
)

call "%VENV_ROOT%\Scripts\activate.bat"
set "MF_LPR_VENV=%VENV_ROOT%"
set "MF_LPR_PYTHON=%VENV_ROOT%\Scripts\python.exe"
set "PYTHONPATH=%~dp0..;%PYTHONPATH%"

echo MF-LPR2 environment active: %VIRTUAL_ENV%
python -c "import torch, numba; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('numba', numba.__version__)"
