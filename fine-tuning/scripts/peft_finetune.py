"""
Hugging Face PEFT Fine-Tuning Script (LoRA / QLoRA)
====================================================
Fine-tunes a causal LM using Hugging Face PEFT (LoRA) and the TRL SFTTrainer.
Does NOT require Unsloth; works with standard Transformers + PEFT + TRL.

Prerequisites:
    Activate virtual environment before running:
        source venv/bin/activate

Usage:
    python scripts/peft_finetune.py

References:
    - https://huggingface.co/docs/peft
    - https://huggingface.co/blog/peft
    - https://huggingface.co/blog/samuellimabraz/peft-methods
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "outputs/peft-lora-adapter"
DATASET_NAME = "mlabonne/FineTome-100k"
MAX_SEQ_LENGTH = 2048
MAX_SAMPLES = 1000
SEED = 42

# LoRA configuration
# r (rank): Controls the capacity of the adapter. Higher rank = more parameters.
#   Typical values: 8, 16, 32, 64.
# lora_alpha: Scaling factor applied to LoRA updates. Often set equal to r.
# lora_dropout: Dropout rate applied inside LoRA layers for regularization.
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

# 4-bit QLoRA quantization (reduces VRAM ~4x vs float32)
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 quantization type
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # Double quantization saves additional VRAM
)

# Training hyperparameters
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    seed=SEED,
    save_steps=50,
    report_to="none",
)


# ---------------------------------------------------------------------------
# 1. Load tokenizer and quantized base model
# ---------------------------------------------------------------------------

print(f"Loading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model in 4-bit: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=QUANTIZATION_CONFIG,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
)
model.config.use_cache = False


# ---------------------------------------------------------------------------
# 2. Wrap model with LoRA adapters
# ---------------------------------------------------------------------------

print("Applying LoRA adapters via PEFT...")
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()
# Output example: trainable params: 6,815,744 || all params: 3,219,914,752 (0.21% trainable)


# ---------------------------------------------------------------------------
# 3. Load dataset
# ---------------------------------------------------------------------------

def format_prompt(sample):
    """Format a single dataset sample into an instruction-response prompt."""
    return {
        "text": (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{sample.get('instruction', '')}\n"
            f"{sample.get('input', '')}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{sample.get('output', '')}<|eot_id|>"
        )
    }


print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")

if MAX_SAMPLES:
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

# If the dataset already has a "conversations" column, use it directly;
# otherwise apply the instruction-response formatter above.
if "conversations" not in dataset.column_names:
    dataset = dataset.map(format_prompt)


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    peft_config=LORA_CONFIG,
    args=TRAINING_ARGS,
)

print("Starting PEFT/LoRA fine-tuning...")
trainer.train()


# ---------------------------------------------------------------------------
# 5. Save adapter
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter weights saved to: {OUTPUT_DIR}")
print("To load at inference time:")
print(f"  from peft import PeftModel")
print(f"  model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")
