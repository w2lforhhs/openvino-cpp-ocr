@echo off
REM PaddleOCR Performance Testing Tool - Windows Setup Script
REM This script creates a Python virtual environment and installs dependencies

echo ========================================
echo PaddleOCR Performance Testing Setup
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python -c "import sys; print(f'Python version: {sys.version}'); exit(0 if sys.version_info >= (3, 8) else 1)"
if %errorlevel% neq 0 (
    echo Error: Python 3.8+ is required
    pause
    exit /b 1
)

echo [2/5] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists, removing old one...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [4/5] Upgrading pip...
python -m pip install --upgrade pip

echo [5/5] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To use the OCR performance tool:
echo 1. Activate the environment: venv\Scripts\activate.bat
echo 2. Run the tool: python ocr_performance.py [image_or_directory]
echo.
echo Examples:
echo   python ocr_performance.py test_image.jpg
echo   python ocr_performance.py images_folder
echo   python ocr_performance.py image.png --lang en --gpu
echo.
echo For help: python ocr_performance.py --help
echo.
pause
