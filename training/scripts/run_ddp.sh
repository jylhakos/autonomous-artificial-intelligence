#!/usr/bin/env bash
# =============================================================================
# run_ddp.sh  --  Distributed pre-training with NativeDDP on 2 GPUs
#
# Engine  : DistributedDataParallel (DDP) via torchrun
# Model   : openai-community/gpt2  (124 M parameters)
# Stage   : pt  (causal language model pre-training on unlabelled text)
# Hardware: Single node, 2x GPU, 12 GB VRAM each
#
# Memory budget per GPU  (GPT-2 124 M, fp16, batch 8 x seq 1024):
#   Model parameters    : ~250 MB
#   Gradients           : ~250 MB
#   Optimizer (fp32 AdamW): ~1 GB
#   Activations         : ~1-2 GB
#   Total approx        : ~3 GB  --  well within 12 GB
#
# FORCE_TORCHRUN=1 instructs llamafactory-cli to launch via torchrun so
# that NCCL-based multi-GPU communication is initialised automatically.
#
# Docs: https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment created by setup.sh if it exists.
# This allows the script to be run without manually sourcing the venv first.
VENV="${SCRIPT_DIR}/.venv/bin/activate"
if [[ -f "${VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}"
fi

CONFIG="configs/pretrain_gpt2_ddp.yaml"

echo "============================================================"
echo "  Distributed Pre-training -- NativeDDP (2 GPUs)"
echo "  Model  : openai-community/gpt2 (124 M)"
echo "  Engine : DistributedDataParallel via torchrun"
echo "  Config : ${CONFIG}"
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

[[ -f "${CONFIG}" ]] || {
    echo "ERROR: Config not found: ${CONFIG}"
    echo "Run this script from the scripts/ directory or check the path."
    exit 1
}

mkdir -p outputs

# CUDA_VISIBLE_DEVICES=0,1 restricts to the first two GPUs explicitly.
# Adjust the indices if your target GPUs are on different device slots.
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train "${CONFIG}"
