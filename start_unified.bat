@echo off
setlocal
pushd "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python or add it to PATH.
    popd
    pause
    exit /b 1
)

python -m server.unified_master
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" (
    echo Unified ApexQuant stopped with exit code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
