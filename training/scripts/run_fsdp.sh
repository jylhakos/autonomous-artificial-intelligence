#!/usr/bin/env bash
# =============================================================================
# run_fsdp.sh  --  Distributed pre-training with FSDP on 2 GPUs
#
# Engine  : FSDP FULL_SHARD via accelerate
# Model   : openai-community/gpt2-xl  (1.5 B parameters)
# Stage   : pt  (causal language model pre-training on unlabelled text)
# Hardware: Single node, 2x GPU, 12 GB VRAM each
#
# FSDP FULL_SHARD shards model parameters, gradients, and optimizer states
# across all GPUs -- equivalent to DeepSpeed ZeRO-3 in memory savings.
# fsdp_offload_params: true in accelerate_fsdp_2gpu.yaml additionally
# offloads parameters to CPU RAM for extra VRAM headroom.
#
# Memory budget per GPU  (GPT-2 XL 1.5 B, fp16, batch 2 x seq 1024):
#   All states sharded   : (3 + 3 + 12) GB / 2 GPUs = ~9 GB
#   Activations          : ~1-2 GB
#   Total approx         : ~10-11 GB  --  fits in 12 GB
#
# Requires LLaMA-Factory to be installed in editable mode (pip install -e .)
# so that src/train.py is accessible.  Run setup.sh first if needed.
# Set LLAMA_FACTORY_DIR to override the default relative path.
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

FSDP_CONFIG="configs/accelerate_fsdp_2gpu.yaml"
TRAIN_CONFIG="configs/pretrain_gpt2xl_fsdp.yaml"

# Locate LLaMA-Factory src/train.py.  setup.sh clones the repo one level up
# from scripts/, i.e. at <workspace>/LLaMA-Factory.
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${SCRIPT_DIR}/../LLaMA-Factory}"
TRAIN_SCRIPT="${LLAMA_FACTORY_DIR}/src/train.py"

echo "============================================================"
echo "  Distributed Pre-training -- FSDP FULL_SHARD (2 GPUs)"
echo "  Model       : openai-community/gpt2-xl (1.5 B)"
echo "  Engine      : FSDP via accelerate (FULL_SHARD + CPU param offload)"
echo "  FSDP config : ${FSDP_CONFIG}"
echo "  Train config: ${TRAIN_CONFIG}"
echo "  Train script: ${TRAIN_SCRIPT}"
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

[[ -f "${FSDP_CONFIG}" ]] || {
    echo "ERROR: Accelerate FSDP config not found: ${FSDP_CONFIG}"
    exit 1
}

[[ -f "${TRAIN_CONFIG}" ]] || {
    echo "ERROR: Training config not found: ${TRAIN_CONFIG}"
    exit 1
}

[[ -f "${TRAIN_SCRIPT}" ]] || {
    echo "ERROR: train.py not found: ${TRAIN_SCRIPT}"
    echo "Set the LLAMA_FACTORY_DIR environment variable or run setup.sh first."
    echo "  export LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory"
    exit 1
}

mkdir -p outputs

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file "${FSDP_CONFIG}" \
    "${TRAIN_SCRIPT}" "${TRAIN_CONFIG}"
