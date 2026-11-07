@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python and enable "Add Python to PATH".
    pause
    exit /b 1
)

python --version
if errorlevel 1 (
    echo Python could not be started.
    pause
    exit /b 1
)

if not exist ".env" (
    echo Missing .env. Create it from .env.example and configure Alpaca paper credentials.
    pause
    exit /b 1
)

python -m server.autonomous_runner
set "RUNNER_EXIT=%ERRORLEVEL%"
if not "%RUNNER_EXIT%"=="0" (
    echo Runner exited with code %RUNNER_EXIT%.
    pause
)
exit /b %RUNNER_EXIT%
