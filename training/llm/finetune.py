"""
finetune.py — Instruction Fine-tuning with PEFT / LoRA
=======================================================
Supervised Fine-Tuning (SFT) of a pre-trained causal LM on instruction data
using Parameter-Efficient Fine-Tuning (LoRA) via the `peft` library.

Workflow:
  1. Load a pre-trained base model (e.g. gpt2, llama, mistral …)
  2. Wrap it with LoRA adapters — only ~0.1–1 % of parameters are trained
  3. Format instruction + response pairs into prompt templates
  4. Train with next-token prediction loss (masked on prompt tokens)
  5. Save merged or adapter-only checkpoints

Supports:
  • HuggingFace Accelerate for multi-GPU / mixed-precision
  • Gradient checkpointing to reduce VRAM
  • Resume from a previous adapter checkpoint

Quick start (single GPU):
    python finetune.py

Multi-GPU (2 GPUs):
    accelerate launch --num_processes 2 finetune.py

Prompt template (Alpaca-style):
    ### Instruction:
    {instruction}

    ### Input:
    {input}   (optional)

    ### Response:
    {output}
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Base model (HuggingFace Hub id or local path)
    "model_name_or_path": "gpt2",
    # Data — use a tiny public instruction dataset by default
    "dataset_name": "tatsu-lab/alpaca",
    "dataset_split": "train",
    "max_samples": 2000,          # set to -1 to use the full split
    # LoRA hyperparameters
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,v_proj",  # comma-separated module names
    # Training
    "output_dir": "./checkpoints/finetune",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_seq_length": 512,
    "bf16": True,
    # Logging
    "logging_steps": 20,
    "save_steps": 200,
    "use_wandb": False,
    "wandb_project": "llm-finetune",
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

ALPACA_PROMPT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca(example: dict) -> str:
    if example.get("input", "").strip():
        return ALPACA_PROMPT.format(**example)
    return ALPACA_PROMPT_NO_INPUT.format(
        instruction=example["instruction"],
        output=example["output"],
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """Tokenised instruction dataset with causal LM labels.

    The prompt portion is masked (label = -100) so the loss is only
    computed on the response tokens.
    """

    def __init__(self, examples: list[str], tokenizer, max_length: int) -> None:
        self.samples: list[dict] = []
        for text in examples:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask padding tokens
            labels[attention_mask == 0] = -100
            self.samples.append(
                {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def build_instruction_dataset(cfg: dict, tokenizer) -> InstructionDataset:
    raw = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])
    if cfg["max_samples"] > 0:
        raw = raw.select(range(min(cfg["max_samples"], len(raw))))
    texts = [format_alpaca(ex) for ex in raw]
    return InstructionDataset(texts, tokenizer, cfg["max_seq_length"])


# ---------------------------------------------------------------------------
# LoRA model
# ---------------------------------------------------------------------------

def build_lora_model(cfg: dict) -> tuple:
    """Return (model, tokenizer) with LoRA adapters applied."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float32,
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in cfg["target_modules"].split(",")]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision="bf16" if cfg["bf16"] else "no",
        log_with="wandb" if cfg["use_wandb"] else None,
        project_dir=cfg["output_dir"],
    )

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
    )

    torch.manual_seed(cfg["seed"])

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", cfg["model_name_or_path"])
    model, tokenizer = build_lora_model(cfg)

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info("Building instruction dataset: %s", cfg["dataset_name"])
    dataset = build_instruction_dataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=2,
    )

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(loader) / cfg["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * cfg["num_train_epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate prepare ───────────────────────────────────────────────────
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    writer = None
    if accelerator.is_main_process:
        out = Path(cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(out / "tensorboard"))
        if cfg["use_wandb"]:
            accelerator.init_trackers(cfg["wandb_project"], config=cfg)

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(cfg["num_train_epochs"]):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{cfg['num_train_epochs']}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.item()

                if accelerator.is_main_process and global_step % cfg["logging_steps"] == 0:
                    avg_loss = epoch_loss / (global_step % (steps_per_epoch + 1) or 1)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "step=%d  loss=%.4f  lr=%.2e", global_step, loss.item(), lr
                    )
                    if writer:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/lr", lr, global_step)

                if accelerator.is_main_process and global_step % cfg["save_steps"] == 0:
                    ckpt_dir = Path(cfg["output_dir"]) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    logger.info("Saved LoRA checkpoint → %s", ckpt_dir)

            progress.set_postfix(loss=f"{loss.item():.4f}")

    # ── Save final adapter ────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(cfg["output_dir"]) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("Fine-tuning complete. Adapter saved to %s", final_dir)
        if writer:
            writer.close()

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM instruction fine-tuning (LoRA / PEFT)")
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", action="store_true", default=val)
        elif val is None:
            p.add_argument(f"--{key}", type=str, default=None)
        else:
            p.add_argument(f"--{key}", type=type(val), default=val)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    train(cfg)


if __name__ == "__main__":
    main()
