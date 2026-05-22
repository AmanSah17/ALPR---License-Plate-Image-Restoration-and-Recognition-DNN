# Activate gemma4 CUDA + Numba environment for MF-LPR2 development.
# Usage (PowerShell):
#   . .\scripts\activate_env.ps1
#   python scripts\verify_phase1.py

$VenvRoot = if ($env:MF_LPR_VENV) { $env:MF_LPR_VENV } else { "D:\gemma4\gemma4" }
$Activate = Join-Path $VenvRoot "Scripts\Activate.ps1"

if (-not (Test-Path $Activate)) {
    Write-Error "Activate script not found: $Activate"
    return
}

. $Activate
$env:MF_LPR_VENV = $VenvRoot
$env:MF_LPR_PYTHON = Join-Path $VenvRoot "Scripts\python.exe"
$env:PYTHONPATH = "$(Resolve-Path (Join-Path $PSScriptRoot '..'));$env:PYTHONPATH"

Write-Host "MF-LPR2 environment active: $env:VIRTUAL_ENV"
Write-Host "Python: $(Get-Command python | Select-Object -ExpandProperty Source)"
python -c "import torch, numba; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('numba', numba.__version__)"
