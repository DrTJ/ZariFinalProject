# Create the Windows virtual environment from requirements.txt.
# Requires Python 3.11 on PATH (the "py" launcher or "python").
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Venv = Join-Path $Root ".venv-windows"
Write-Host "Creating .venv-windows from requirements.txt."
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 -m venv $Venv
} else {
    & python -m venv $Venv
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Ready. Run: .\run.ps1 --video data/vtest.avi --classes person --modes cascade --track --save-video"
