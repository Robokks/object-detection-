@echo off
REM Builds PinDetector.exe from the desktop app. Double-click this file —
REM no typing needed. Needs the backend\.venv virtual environment already
REM set up (the same one used by the "Backend (FastAPI)" PyCharm run
REM configuration); if you haven't created it yet, do that first via
REM PyCharm: Settings > Project > Python Interpreter > Add Interpreter >
REM Virtualenv > New, pointed at backend\.venv.

setlocal

cd /d "%~dp0"

if not exist "..\backend\.venv\Scripts\python.exe" (
    echo Could not find ..\backend\.venv\Scripts\python.exe
    echo Set up that virtual environment first, then run this again.
    pause
    exit /b 1
)

set PY=..\backend\.venv\Scripts\python.exe

echo Installing required packages, including PyInstaller...
echo (first run only - this can take a few minutes)
"%PY%" -m pip install -q -r requirements.txt
"%PY%" -m pip install -q pyinstaller
if errorlevel 1 (
    echo.
    echo Package install failed - see the errors above.
    pause
    exit /b 1
)

echo.
echo Building PinDetector.exe - this bundles PySide6, PyTorch, and
echo Ultralytics together, so it can take several minutes. Please wait...
"%PY%" -m PyInstaller pin_detector.spec --noconfirm
if errorlevel 1 (
    echo.
    echo Build failed - see the errors above.
    pause
    exit /b 1
)

echo.
echo Done. PinDetector.exe is in desktop\dist\PinDetector\
echo Zip that whole PinDetector folder to move it to another machine -
echo the .exe needs the rest of the folder next to it.
pause
