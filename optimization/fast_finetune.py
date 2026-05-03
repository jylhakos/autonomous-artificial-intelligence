"""
fast_finetune.py
================
Demonstrates fast LoRA fine-tuning with Unsloth, which patches Hugging Face
Transformers to deliver up to 2x faster training and significantly lower VRAM
consumption through kernel-level optimizations.

README section: "Using Unsloth for Fast Fine-Tuning"

Concept covered:
    Unsloth implements custom Triton CUDA kernels for RoPE embeddings, cross-
    entropy loss, and Flash Attention 2, bypassing the standard PyTorch
    autograd graph for these operations.  The result is faster backward passes
    and lower peak activation memory.

    Combined with QLoRA (4-bit base model + LoRA adapters), Unsloth allows
    fine-tuning Llama-3-8B on a single 16 GB VRAM GPU.  This script sets up
    the model with:
    - 4-bit NF4 quantization for the frozen base model
    - LoRA adapters injected into Q/K/V/O projections (rank r=16)
    - Gradient checkpointing via Unsloth's efficient implementation

Related README sections:
    - "Parameter-Efficient Fine-Tuning (PEFT) / LoRA"  — LoRA mathematics
    - "Fine-Tuning Language Models"                    — memory challenge
    - "Fine-Tuning Methods" table                      — QLoRA memory cost

Requirements:
    - NVIDIA GPU with CUDA (minimum 16 GB VRAM recommended for 8B models)
    - Virtual environment activated (see README "Python Virtual Environment Setup")

    # Install Unsloth (CUDA-specific wheel; adjust for your CUDA version)
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install torch transformers peft

Usage:
    # 1. Activate the virtual environment
    source .venv/bin/activate

    # 2. Run the script
    python fast_finetune.py
"""

from unsloth import FastLanguageModel
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # Pre-quantized 4-bit model from Unsloth
MAX_SEQ_LENGTH = 2048                         # Maximum sequence length for training
LORA_RANK = 16                                # LoRA rank r (higher = more capacity)
LORA_ALPHA = 16                               # LoRA scaling factor alpha

# ---------------------------------------------------------------------------
# Load model in 4-bit precision (QLoRA base)
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_NAME} in 4-bit precision ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,        # Auto-detect: BF16 on Ampere+, FP16 otherwise
    load_in_4bit=True,
)

# ---------------------------------------------------------------------------
# Inject LoRA adapters
# Only the adapter matrices A and B are trainable; the base model is frozen.
# target_modules selects which weight matrices to adapt.
# ---------------------------------------------------------------------------
print(f"Adding LoRA adapters (rank={LORA_RANK}) ...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,      # Unsloth optimizes for dropout=0
    bias="none",
    use_gradient_checkpointing="unsloth",  # Smart gradient checkpointing
    random_state=42,
)

# ---------------------------------------------------------------------------
# Report trainable parameters
# ---------------------------------------------------------------------------
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ---------------------------------------------------------------------------
# Placeholder training step
# Replace this section with a real Hugging Face Trainer or trl.SFTTrainer call
# using your own dataset.
# ---------------------------------------------------------------------------
print("\nModel is ready for fine-tuning.")
print("To train, configure a trl.SFTTrainer with your dataset and call trainer.train().")
print("Example:")
print("    from trl import SFTTrainer")
print("    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, ...)")
print("    trainer.train()")
