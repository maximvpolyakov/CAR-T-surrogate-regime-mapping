@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.14 3.13 3.12 3.11 3.10) do (
        if not defined PY_CMD (
            py -%%V -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
            if not errorlevel 1 set "PY_CMD=py -%%V"
        )
    )
)
if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    echo Python 3.10 or newer was not found.
    echo Install Python 3.10 or newer and run this file again.
    pause
    exit /b 1
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if errorlevel 1 rmdir /s /q ".venv"
)
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    %PY_CMD% -m venv --system-site-packages .venv
    if errorlevel 1 (
        echo Failed to create the Python environment.
        pause
        exit /b 2
    )
)
call ".venv\Scripts\activate.bat"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "PIP_PROXY="
set "PIP_CONFIG_FILE=NUL"
set "PYTHONPATH=%CD%\src"
echo.
echo Python environment:
python --version
python -c "import numpy, scipy, pandas, sklearn, matplotlib, yaml, joblib" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Installing required Python packages...
    python -m pip install --disable-pip-version-check numpy scipy pandas scikit-learn matplotlib PyYAML joblib
    if errorlevel 1 (
        echo.
        echo Package installation failed. Check the Internet connection or proxy settings.
        pause
        exit /b 3
    )
)
set "OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
echo.
echo Starting the study...
python -m cart_pipeline --config configs\study.yaml --output results
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The study stopped with error code %RC%.
    pause
)
endlocal & exit /b %RC%
