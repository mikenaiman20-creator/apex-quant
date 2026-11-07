@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python and enable "Add Python to PATH".
    pause
    exit /b 1
)

python -m server.dashboard
set "DASHBOARD_EXIT=%ERRORLEVEL%"
if not "%DASHBOARD_EXIT%"=="0" (
    echo Dashboard exited with code %DASHBOARD_EXIT%.
    pause
)
exit /b %DASHBOARD_EXIT%
