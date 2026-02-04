@echo off
echo ================================================
echo   Physics Application Suite v2.0 - PyQt5
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.11 or later.
    pause
    exit /b 1
)

REM Install PyQt5 and other requirements
echo Installing PyQt5 and dependencies...
pip install PyQt5 matplotlib numpy pillow

REM Launch the PyQt5 application
echo.
echo 🚀 Launching Physics Application Suite...
echo.
python main.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Application encountered an error.
    echo Check that all dependencies are properly installed.
)

pause