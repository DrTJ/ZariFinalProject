# Run the project with the Windows virtual environment.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root ".venv-windows\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Windows environment is missing. Run setup_windows.ps1 first."
}
& $Python -m cascade_mot.run @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
