#!/usr/bin/env bash
# =============================================================================
# run_deepspeed_z2.sh  --  Distributed pre-training with DeepSpeed ZeRO-2
#
# Engine  : DeepSpeed ZeRO stage 2
# Model   : openai-community/gpt2-xl  (1.5 B parameters)
# Stage   : pt  (causal language model pre-training on unlabelled text)
# Hardware: Single node, 2x GPU, 12 GB VRAM each
#
# ZeRO-2 shards optimizer states and gradients across both GPUs.
# Each GPU still holds a full copy of model parameters.
#
# Memory budget per GPU  (GPT-2 XL 1.5 B, bf16, batch 2 x seq 1024):
#   Model parameters    : ~3 GB  (full copy on each GPU)
#   Gradients (sharded) : ~1.5 GB  (3 GB / 2 GPUs)
#   Optimizer (sharded) : ~6 GB  (12 GB / 2 GPUs)
#   Activations         : ~1-2 GB
#   Total approx        : ~11-12 GB  --  fits in 12 GB with grad checkpointing
#
# Docs: https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment created by setup.sh if it exists.
VENV="${SCRIPT_DIR}/.venv/bin/activate"
if [[ -f "${VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}"
fi

CONFIG="configs/pretrain_gpt2xl_ds_z2.yaml"

echo "============================================================"
echo "  Distributed Pre-training -- DeepSpeed ZeRO-2 (2 GPUs)"
echo "  Model  : openai-community/gpt2-xl (1.5 B)"
echo "  Engine : DeepSpeed ZeRO stage 2  (shards optimizer + gradients)"
echo "  Config : ${CONFIG}"
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

[[ -f "${CONFIG}" ]] || {
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
}

mkdir -p outputs

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train "${CONFIG}"
