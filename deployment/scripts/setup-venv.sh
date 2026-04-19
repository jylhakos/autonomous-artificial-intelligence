#!/bin/bash

# Setup script for Python virtual environment
# This script creates and configures a virtual environment for AI agent development

set -e  # Exit on error

echo "=========================================="
echo "AI Agent Virtual Environment Setup"
echo "=========================================="
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*|MINGW*|MSYS*) MACHINE=Windows;;
    *)          MACHINE="UNKNOWN"
esac

echo "Detected OS: ${MACHINE}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

echo "Python version: ${PYTHON_VERSION}"

if ! printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V -C; then
    echo "Error: Python 3.10 or higher is required"
    exit 1
fi

echo "Python version check: OK"
echo ""

# Create virtual environment
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created: $VENV_DIR"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."

if [ "$MACHINE" = "Windows" ]; then
    ACTIVATE_CMD="$VENV_DIR/Scripts/activate"
else
    ACTIVATE_CMD="source $VENV_DIR/bin/activate"
fi

# Upgrade pip
echo "Upgrading pip..."
if [ "$MACHINE" = "Windows" ]; then
    ./$VENV_DIR/Scripts/python -m pip install --upgrade pip
else
    ./$VENV_DIR/bin/python -m pip install --upgrade pip
fi

echo ""

# Install requirements if requirements.txt exists
if [ -f "scripts/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    if [ "$MACHINE" = "Windows" ]; then
        ./$VENV_DIR/Scripts/pip install -r scripts/requirements.txt
    else
        ./$VENV_DIR/bin/pip install -r scripts/requirements.txt
    fi
    echo "Dependencies installed successfully"
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo ""

if [ "$MACHINE" = "Windows" ]; then
    echo "  venv\\Scripts\\activate"
else
    echo "  source venv/bin/activate"
fi

echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To install specific platform dependencies:"
echo "  pip install -r scripts/requirements.txt"
echo ""
