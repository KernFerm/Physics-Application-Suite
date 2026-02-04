#!/bin/bash

echo "================================================"
echo "  Physics Application Suite v2.0 - PyQt5"
echo "================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.11 or later."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3."
    exit 1
fi

# Install PyQt5 and other requirements
echo "Installing PyQt5 and dependencies..."
pip3 install PyQt5 matplotlib numpy pillow

# Launch the PyQt5 application
echo
echo "🚀 Launching Physics Application Suite..."
echo
python3 main.py

if [ $? -ne 0 ]; then
    echo
    echo "❌ Application encountered an error."
    echo "Check that all dependencies are properly installed."
fi
