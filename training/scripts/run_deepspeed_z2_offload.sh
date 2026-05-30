#!/usr/bin/env bash
# =============================================================================
# run_deepspeed_z2_offload.sh  --  DeepSpeed ZeRO-2 + CPU offload on 2 GPUs
#
# Engine  : DeepSpeed ZeRO stage 2 with optimizer CPU offload
# Model   : openai-community/gpt2-xl  (1.5 B parameters)
# Stage   : pt  (causal language model pre-training on unlabelled text)
# Hardware: Single node, 2x GPU, 12 GB VRAM each
#
# ZeRO-2 + CPU offload moves optimizer states from GPU VRAM to system RAM.
# Use this variant when pure ZeRO-2 still approaches the 12 GB VRAM limit,
# e.g. when increasing batch size or context length.
#
# Trade-off:
#   VRAM freed   : optimizer state tensors (~6 GB sharded) transferred to RAM
#   Speed cost   : 20-40% slower than ZeRO-2 without offload due to PCIe
#   RAM required : >= 32 GB system RAM recommended
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

CONFIG="configs/pretrain_gpt2xl_ds_z2_offload.yaml"

echo "============================================================"
echo "  Distributed Pre-training -- DeepSpeed ZeRO-2 + CPU Offload"
echo "  Model  : openai-community/gpt2-xl (1.5 B)"
echo "  Engine : DeepSpeed ZeRO-2 with optimizer offloaded to CPU"
echo "  Config : ${CONFIG}"
echo "  Note   : Requires >= 32 GB system RAM for CPU offload buffers"
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

[[ -f "${CONFIG}" ]] || {
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
}

mkdir -p outputs

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train "${CONFIG}"
