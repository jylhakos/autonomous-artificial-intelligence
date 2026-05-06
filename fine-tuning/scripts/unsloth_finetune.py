"""
Unsloth Fine-Tuning Script
==========================
Fine-tunes an open-source LLM (e.g., Llama 3.1 8B) using QLoRA via Unsloth.

Prerequisites:
    Activate virtual environment before running:
        source venv/bin/activate

Usage:
    python scripts/unsloth_finetune.py

References:
    - https://github.com/unslothai/unsloth
    - https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
"""

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model and adapter settings
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048       # Context window: maximum number of tokens per sample
DTYPE = None                # Auto-detect: float16 for older GPUs, bfloat16 for Ampere+
LOAD_IN_4BIT = True         # Enable 4-bit QLoRA quantization to reduce VRAM usage

# LoRA hyperparameters
LORA_RANK = 16              # Rank r: lower = fewer parameters; 8 or 16 is a common starting point
LORA_ALPHA = 16             # Scaling factor; typically set equal to rank
LORA_DROPOUT = 0.0          # 0 is optimal for Unsloth; non-zero adds regularization
LORA_BIAS = "none"          # Do not train bias terms

# Training hyperparameters
LEARNING_RATE = 2e-4        # Step size for weight updates
BATCH_SIZE = 2              # Samples processed per step
GRAD_ACCUMULATION_STEPS = 4 # Accumulate gradients to simulate a larger batch
WARMUP_STEPS = 5            # Steps to linearly increase LR from 0 before main schedule
MAX_STEPS = 60              # Total training steps (set to -1 to train for full epochs)
NUM_EPOCHS = 1              # Number of full passes over the dataset (-1 if using MAX_STEPS)
WEIGHT_DECAY = 0.01         # L2 regularization to reduce overfitting
LR_SCHEDULER = "linear"     # Learning rate schedule type
SEED = 42                   # Reproducibility seed

# Temperature (inference only, not used during training)
# TEMPERATURE controls randomness in the model's output distribution.
# Low (0.1-0.3): deterministic, focused, factual answers.
# Medium (0.7):  balanced creativity and coherence.
# High (1.0+):   diverse, creative, but potentially incoherent.
TEMPERATURE = 0.7

# Output directory
OUTPUT_DIR = "outputs/unsloth-llama-lora"

# Dataset settings
DATASET_NAME = "mlabonne/FineTome-100k"
DATASET_SPLIT = "train"
MAX_SAMPLES = 1000          # Limit samples for a quick test; remove for full training


# ---------------------------------------------------------------------------
# 1. Load model and tokenizer
# ---------------------------------------------------------------------------

print(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# Apply chat template so the tokenizer formats prompts correctly
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")


# ---------------------------------------------------------------------------
# 2. Apply LoRA adapters (PEFT)
# ---------------------------------------------------------------------------

print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    use_gradient_checkpointing="unsloth",   # Unsloth-optimized checkpointing
    random_state=SEED,
    use_rslora=True,        # Rank-Stabilized LoRA: scales by 1/sqrt(r) for stability
)

model.print_trainable_parameters()


# ---------------------------------------------------------------------------
# 3. Load and prepare dataset
# ---------------------------------------------------------------------------

def format_conversations(examples):
    """Format dataset samples into the model's chat template."""
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}


print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

if MAX_SAMPLES:
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

dataset = dataset.map(format_conversations, batched=True)
print(f"Dataset size: {len(dataset)} samples")


# ---------------------------------------------------------------------------
# 4. Configure and run trainer
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    num_train_epochs=NUM_EPOCHS if MAX_STEPS == -1 else 1,
    learning_rate=LEARNING_RATE,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=LR_SCHEDULER,
    seed=SEED,
    report_to="none",       # Change to "wandb" to log metrics to Weights & Biases
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    args=training_args,
)

print("Starting training...")
trainer_stats = trainer.train()
print(f"Training complete. Stats: {trainer_stats}")


# ---------------------------------------------------------------------------
# 5. Save LoRA adapter weights
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# 6. Optional: Merge adapter and save as full model (for deployment)
# ---------------------------------------------------------------------------

# Uncomment to merge LoRA weights into the base model for standalone deployment:
# model.save_pretrained_merged(
#     "outputs/unsloth-llama-merged",
#     tokenizer,
#     save_method="merged_16bit",
# )

# Uncomment to export as GGUF format for llama.cpp / Ollama:
# model.save_pretrained_gguf(
#     "outputs/unsloth-llama-gguf",
#     tokenizer,
#     quantization_method="q4_k_m",
# )


# ---------------------------------------------------------------------------
# 7. Inference example (greedy, temperature-controlled)
# ---------------------------------------------------------------------------

print("\nRunning inference example...")
FastLanguageModel.for_inference(model)  # Enable 2x faster inference

messages = [
    {"role": "user", "content": "Explain the difference between LoRA and full fine-tuning."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=TEMPERATURE,
    do_sample=True,         # Set False for greedy (deterministic) decoding
    top_p=0.9,
    repetition_penalty=1.1,
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"Model response:\n{response}")
