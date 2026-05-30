#!/usr/bin/env bash
# =============================================================================
# setup.sh  --  Install LLaMA-Factory for distributed pre-training
#
# Target  : Ubuntu x86_64, Python 3.11+, CUDA 12.2+
# Hardware: Single node, 2x GPU, 12 GB VRAM each
#
# What this script does:
#   1. Verifies Python 3.11+ and CUDA / GPU availability
#   2. Creates a Python virtual environment in scripts/.venv
#   3. Clones LLaMA-Factory one level above scripts/ (at <workspace>/LLaMA-Factory)
#   4. Installs LLaMA-Factory in editable mode plus dependencies
#   5. Installs DeepSpeed (required for ZeRO-2 / ZeRO-3 scripts)
#   6. Optionally installs Flash Attention 2 (Ampere+ GPUs only)
#
# After setup, activate the environment and run any training script:
#   source scripts/.venv/bin/activate
#   bash scripts/run_ddp.sh
#
# Docs: https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LLAMA_FACTORY_DIR="${SCRIPT_DIR}/../LLaMA-Factory"

echo "============================================================"
echo "  LLaMA-Factory Setup for Distributed Pre-training"
echo "  Target  : Ubuntu x86_64, 2x GPU (12 GB VRAM each)"
echo "============================================================"
echo ""

# ---- 1. Python version check ------------------------------------------------
echo "[1/6] Checking Python version (3.11+ required)..."
python3 --version
python3 - <<'EOF'
import sys
if sys.version_info < (3, 11):
    print(f"ERROR: Python 3.11+ required, found {sys.version}")
    sys.exit(1)
print("Python version OK")
EOF

# ---- 2. GPU / CUDA check ----------------------------------------------------
echo ""
echo "[2/6] Checking GPU and CUDA..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | \
        awk -F',' '{printf "  GPU %s: %s (%s)\n", $1, $2, $3}'
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "  Found ${GPU_COUNT} GPU(s)"
    if [[ "${GPU_COUNT}" -lt 2 ]]; then
        echo "  WARNING: At least 2 GPUs are required for distributed training"
    fi
else
    echo "  WARNING: nvidia-smi not found -- ensure NVIDIA drivers are installed"
fi

if command -v nvcc &>/dev/null; then
    nvcc --version | grep "release"
else
    echo "  WARNING: nvcc not found -- ensure CUDA 12.2+ is installed and in PATH"
fi

# ---- 3. Virtual environment --------------------------------------------------
echo ""
echo "[3/6] Setting up Python virtual environment..."
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created : ${VENV_DIR}"
else
    echo "  Exists  : ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
echo "  pip upgraded"

# ---- 4. Clone LLaMA-Factory --------------------------------------------------
echo ""
echo "[4/6] Cloning LLaMA-Factory..."
if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LLAMA_FACTORY_DIR}"
    echo "  Cloned to : ${LLAMA_FACTORY_DIR}"
else
    echo "  Exists    : ${LLAMA_FACTORY_DIR}"
fi

# ---- 5. Install LLaMA-Factory and dependencies -------------------------------
echo ""
echo "[5/6] Installing LLaMA-Factory and dependencies..."
cd "${LLAMA_FACTORY_DIR}"

# Editable install keeps src/train.py accessible for the FSDP run script.
pip install -e "." -r requirements/metrics.txt --quiet
echo "  LLaMA-Factory installed (editable)"

# DeepSpeed is required for the ZeRO-2 and ZeRO-3 run scripts.
pip install deepspeed --quiet
echo "  DeepSpeed installed"

# Flash Attention 2 improves throughput on Ampere+ GPUs (optional).
# Build requires a matching CUDA toolkit and C++ compiler.
pip install flash-attn --no-build-isolation --quiet 2>/dev/null \
    || echo "  INFO: flash-attn skipped (optional -- requires Ampere+ GPU and matching CUDA headers)"

# ---- 6. Verify ---------------------------------------------------------------
echo ""
echo "[6/6] Verifying installation..."
llamafactory-cli version

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Activate environment:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  Run distributed pre-training (from scripts/ directory):"
echo "    bash run_ddp.sh                   # NativeDDP,    GPT-2 (124 M)"
echo "    bash run_deepspeed_z2.sh          # ZeRO-2,       GPT-2 XL (1.5 B)"
echo "    bash run_deepspeed_z2_offload.sh  # ZeRO-2+CPU,   GPT-2 XL (1.5 B)"
echo "    bash run_deepspeed_z3_offload.sh  # ZeRO-3+CPU,   GPT-2 XL (1.5 B)"
echo "    bash run_fsdp.sh                  # FSDP,         GPT-2 XL (1.5 B)"
echo "============================================================"
