"""
DPO (Direct Preference Optimization) Alignment Script for LLMs.

This script demonstrates how to align an open-source Large Language Model (LLM)
like LLaMA for security using PyTorch and Hugging Face. It implements Direct
Preference Optimization (DPO) to teach the model to refuse harmful prompts
(e.g., malware generation, cyberattacks) while remaining helpful for safe queries.

Alignment Workflow:
    1. Supervised Fine-Tuning (SFT): Train the model on safe, curated prompt-response pairs.
    2. Preference Dataset Creation: Gather pairs of "chosen" (safe/refusal) and
       "rejected" (harmful/unsafe) model responses.
    3. Alignment Optimization: Use DPO to update model weights based on preferences.

Dataset format (PKU-SafeRLHF triplets):
    - prompt:          The user query.
    - chosen:          The safe, aligned model response.
    - rejected:        The unsafe, unaligned model response.

Requirements:
    pip install torch transformers datasets trl

Usage:
    python scripts/dpo_alignment.py

References:
    - DPO paper: https://arxiv.org/abs/2305.18290
    - PKU-SafeRLHF dataset: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
    - TRL library: https://github.com/huggingface/trl
    - AWS Blog: https://aws.amazon.com/blogs/machine-learning/an-introduction-to-preparing-your-own-dataset-for-llm-training/
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

# ---------------------------------------------------------------------------
# 1. Load the base model and tokenizer (e.g., LLaMA-3)
# ---------------------------------------------------------------------------
model_id = "meta-llama/Meta-Llama-3-8B"
device_map = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

# Reference model for DPO (keeps the fine-tuned model from drifting too far
# from the original distribution — this is the core of the DPO loss function).
ref_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

# ---------------------------------------------------------------------------
# 2. Load a security alignment dataset
#    Dataset must contain columns: 'prompt', 'chosen' (safe), 'rejected' (unsafe)
# ---------------------------------------------------------------------------
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train[:1000]")


def format_dpo_dataset(example: dict) -> dict:
    """
    Map the PKU-SafeRLHF dataset schema to the DPO triplet format
    expected by the TRL DPOTrainer.

    Args:
        example: A single dataset row containing the original fields.

    Returns:
        A dictionary with 'prompt', 'chosen', and 'rejected' keys.
    """
    return {
        "prompt": example["prompt"],
        "chosen": example["response_safe"],
        "rejected": example["response_unsafe"]
    }


formatted_dataset = dataset.map(format_dpo_dataset)

# ---------------------------------------------------------------------------
# 3. Configure training arguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,      # Effective batch size = 2 * 4 = 8
    learning_rate=5e-7,                 # Conservative LR to avoid over-fitting
    logging_steps=10,
    output_dir="./secure_llama_dpo",
    bf16=True,                          # Use bfloat16 for modern GPUs (A100, H100)
    remove_unused_columns=False,
    gradient_checkpointing=True         # Trades compute for VRAM savings
)

# ---------------------------------------------------------------------------
# 4. Initialize and run the DPO Trainer
# ---------------------------------------------------------------------------
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,                           # Controls strength of alignment vs. base model:
                                        # lower beta = stronger alignment push,
                                        # higher beta = closer to original distribution.
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_length=512,
    max_prompt_length=256,
)

print("Starting security alignment training via DPO...")
dpo_trainer.train()

# ---------------------------------------------------------------------------
# 5. Save the aligned, secure model
# ---------------------------------------------------------------------------
model.save_pretrained("./secure_llama_final")
tokenizer.save_pretrained("./secure_llama_final")

print("Alignment complete. Aligned model saved to ./secure_llama_final")
