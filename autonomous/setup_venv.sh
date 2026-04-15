#!/bin/bash

# Virtual Environment Setup Script for VS Code and Linux
# This script automates the setup of the Python virtual environment

set -e  # Exit on error

PROJECT_DIR="/home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/autonomous"
VENV_DIR="$PROJECT_DIR/venv"

echo "========================================="
echo "Virtual Environment Setup"
echo "========================================="
echo ""

# Step 1: Check if Python 3 is installed
echo "Step 1: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "✓ Found: $PYTHON_VERSION"
echo ""

# Step 2: Navigate to project directory
echo "Step 2: Navigating to project directory..."
cd "$PROJECT_DIR"
echo "✓ Current directory: $(pwd)"
echo ""

# Step 3: Check if virtual environment exists
if [ -d "$VENV_DIR" ]; then
    echo "Step 3: Virtual environment already exists"
    echo "✓ Found: $VENV_DIR"
else
    echo "Step 3: Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created: $VENV_DIR"
fi
echo ""

# Step 4: Activate virtual environment
echo "Step 4: Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Step 5: Upgrade pip
echo "Step 5: Upgrading pip..."
pip install --upgrade pip --quiet
PIP_VERSION=$(pip --version)
echo "✓ $PIP_VERSION"
echo ""

# Step 6: Install dependencies
echo "Step 6: Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "✓ Dependencies installed"
else
    echo "⚠ Warning: requirements.txt not found"
fi
echo ""

# Step 7: Verify installation
echo "Step 7: Verifying installation..."
echo "Installed packages:"
pip list | head -n 20
echo "... (showing first 20 packages)"
echo ""

# Step 8: Create .env file if it doesn't exist
echo "Step 8: Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "⚠ Please edit .env and add your API keys"
else
    echo "✓ .env file exists"
fi
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Virtual environment is active and ready to use."
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To run examples:"
echo "  python examples/simple_agent.py"
echo "  python examples/multi_agent_system.py"
echo "  streamlit run examples/multi_model_chat.py"
echo ""
echo "Don't forget to configure your API keys in .env file!"
echo "========================================="
