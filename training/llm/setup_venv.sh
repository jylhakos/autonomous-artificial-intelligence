#!/usr/bin/env bash
# setup_venv.sh — Create and activate a Python virtual environment for LLM training
# Usage:
#   bash setup_venv.sh          # create venv and install requirements
#   source setup_venv.sh        # create + activate in the current shell
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "==> Checking Python version..."
$PYTHON --version

echo "==> Creating virtual environment in '${VENV_DIR}'..."
$PYTHON -m venv "$VENV_DIR"

echo "==> Activating virtual environment..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch (CPU build by default)..."
# For CUDA 12.1 replace with:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing remaining requirements..."
pip install -r requirements.txt

echo ""
echo "✓  Virtual environment ready."
echo "   Activate with:  source ${VENV_DIR}/bin/activate"
echo "   Deactivate with: deactivate"
