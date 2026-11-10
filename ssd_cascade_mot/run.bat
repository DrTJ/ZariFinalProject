@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-windows\Scripts\python.exe" (
  echo Windows runtime is missing. Run setup_windows.bat first.
  exit /b 1
)
".venv-windows\Scripts\python.exe" "%~dp0run.py" %*
