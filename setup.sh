#!/bin/bash
# PaddleOCR Performance Testing Tool - Linux/macOS Setup Script
# This script creates a Python virtual environment and installs dependencies

set -e  # Exit on any error

echo "========================================"
echo "PaddleOCR Performance Testing Setup"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ using your package manager"
    exit 1
fi

echo "[1/5] Checking Python version..."
python3 -c "import sys; print(f'Python version: {sys.version}'); exit(0 if sys.version_info >= (3, 8) else 1)"

echo "[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, removing old one..."
    rm -rf venv
fi

python3 -m venv venv

echo "[3/5] Activating virtual environment..."
source venv/bin/activate

echo "[4/5] Upgrading pip..."
python -m pip install --upgrade pip

echo "[5/5] Installing dependencies..."
pip install -r requirements.txt

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "To use the OCR performance tool:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the tool: python ocr_performance.py [image_or_directory]"
echo
echo "Examples:"
echo "  python ocr_performance.py test_image.jpg"
echo "  python ocr_performance.py images_folder"
echo "  python ocr_performance.py image.png --lang en --gpu"
echo
echo "For help: python ocr_performance.py --help"
echo
