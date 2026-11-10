# Windows entry. Uses .venv-windows and never the Mac .venv.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root ".venv-windows\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Windows runtime is missing. Run setup_windows.ps1 first."
}
& $Python (Join-Path $Root "run.py") @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
