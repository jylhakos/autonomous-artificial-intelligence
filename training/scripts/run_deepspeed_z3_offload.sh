#!/usr/bin/env bash
# =============================================================================
# run_deepspeed_z3_offload.sh  --  DeepSpeed ZeRO-3 + full CPU offload on 2 GPUs
#
# Engine  : DeepSpeed ZeRO stage 3 with full CPU offload (params + optimizer)
# Model   : openai-community/gpt2-xl  (1.5 B parameters)
# Stage   : pt  (causal language model pre-training on unlabelled text)
# Hardware: Single node, 2x GPU, 12 GB VRAM each
#
# ZeRO-3 shards ALL tensors across GPUs: model parameters, gradients, and
# optimizer states.  Adding CPU offload further moves sharded parameters and
# optimizer states to system RAM, maximising the model size that can be trained
# on limited VRAM.
#
# Use this when ZeRO-2 cannot fit the model, or when scaling to larger
# architectures (e.g. 7B+ parameters) on the same 2x 12 GB hardware.
#
# Trade-off:
#   VRAM freed   : most tensors moved to CPU
#   Speed cost   : significant throughput reduction (PCIe bottleneck)
#   RAM required : >= 64 GB system RAM recommended
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

CONFIG="configs/pretrain_gpt2xl_ds_z3_offload.yaml"

echo "============================================================"
echo "  Distributed Pre-training -- DeepSpeed ZeRO-3 + CPU Offload"
echo "  Model  : openai-community/gpt2-xl (1.5 B)"
echo "  Engine : DeepSpeed ZeRO-3 with full CPU offload (params + optimizer)"
echo "  Config : ${CONFIG}"
echo "  Note   : Requires >= 64 GB system RAM.  Throughput is reduced."
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

[[ -f "${CONFIG}" ]] || {
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
}

mkdir -p outputs

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train "${CONFIG}"
