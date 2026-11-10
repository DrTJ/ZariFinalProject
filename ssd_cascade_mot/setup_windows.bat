@echo off
setlocal
cd /d "%~dp0"
echo Creating .venv-windows from requirements.txt.
where py >nul 2>&1
if %errorlevel%==0 (
  py -3.11 -m venv .venv-windows
) else (
  python -m venv .venv-windows
)
if errorlevel 1 exit /b 1
".venv-windows\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv-windows\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 exit /b 1
echo Ready. Run: run.bat --video data\vtest.avi --classes person --modes cascade --track --save-video
