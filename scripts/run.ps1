# Run any project script with the gemma4 interpreter (no manual activate needed).
# Usage:
#   .\scripts\run.ps1 scripts\verify_phase1.py --profile train

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = if ($env:MF_LPR_PYTHON) { $env:MF_LPR_PYTHON } else { "D:\gemma4\gemma4\Scripts\python.exe" }

if (-not (Test-Path $Python)) {
    Write-Error "Python not found: $Python. Set MF_LPR_PYTHON or install gemma4 venv."
    exit 1
}

$ScriptPath = Join-Path $ProjectRoot $Script
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

& $Python $ScriptPath @Args
exit $LASTEXITCODE
