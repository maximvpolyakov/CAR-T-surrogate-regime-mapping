@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run run_study.bat once before rebuilding figures.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
set "PYTHONPATH=%CD%\src"
python -m cart_pipeline.replot --results results --output results\figures_rebuilt
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo Figure rebuilding stopped with error code %RC%.
    pause
)
endlocal & exit /b %RC%
