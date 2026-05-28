"""
pretrain.py — GPT-style Causal Language Model Pre-training
===========================================================
Trains a decoder-only Transformer from scratch using next-token prediction
(causal language modelling / CLM).

Supports:
  • Single GPU training
  • Multi-GPU via DDP (torchrun)
  • Optional gradient accumulation, mixed-precision (BF16/FP16)
  • Linear warm-up + cosine LR decay
  • Periodic checkpointing and resumption
  • WandB or TensorBoard logging

Quick start (single GPU):
    python pretrain.py --config configs/pretrain_small.json

Multi-GPU (2 GPUs, DDP):
    torchrun --nproc_per_node=2 pretrain.py --config configs/pretrain_small.json

Example minimal config (JSON):
{
    "model_name_or_path": null,          # null = train from scratch
    "vocab_size": 32000,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "max_position_embeddings": 1024,
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "tokenizer_name": "gpt2",
    "output_dir": "./checkpoints/pretrain",
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "learning_rate": 3e-4,
    "warmup_steps": 1000,
    "weight_decay": 0.1,
    "fp16": false,
    "bf16": true,
    "logging_steps": 100,
    "save_steps": 1000,
    "use_wandb": false
}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters (override via JSON config)
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Model
    "model_arch": "gpt2",              # "gpt2" | "llama"
    "model_name_or_path": None,        # None = train from scratch
    "vocab_size": 50257,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 1024,
    "max_position_embeddings": 512,
    # Data
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "tokenizer_name": "gpt2",
    "block_size": 512,
    # Training
    "output_dir": "./checkpoints/pretrain",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 1,
    "max_steps": -1,
    "learning_rate": 3e-4,
    "warmup_steps": 200,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "fp16": False,
    "bf16": False,
    # Logging / saving
    "logging_steps": 50,
    "save_steps": 500,
    "use_wandb": False,
    "wandb_project": "llm-pretrain",
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialise DDP if running under torchrun; return (rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> torch.nn.Module:
    """Build a fresh Transformer model from config or load a checkpoint."""
    if cfg["model_name_or_path"]:
        if cfg["model_arch"] == "llama":
            model = LlamaForCausalLM.from_pretrained(cfg["model_name_or_path"])
        else:
            model = GPT2LMHeadModel.from_pretrained(cfg["model_name_or_path"])
        return model

    if cfg["model_arch"] == "llama":
        model_cfg = LlamaConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg["max_position_embeddings"],
        )
        return LlamaForCausalLM(model_cfg)

    # Default: GPT-2 style
    model_cfg = GPT2Config(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["hidden_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_inner=cfg["intermediate_size"],
        n_positions=cfg["max_position_embeddings"],
    )
    return GPT2LMHeadModel(model_cfg)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def tokenize_and_chunk(examples: dict, tokenizer, block_size: int) -> dict:
    """Tokenise a batch of texts and concatenate into fixed-size blocks."""
    all_ids: list[int] = []
    for text in examples["text"]:
        all_ids.extend(tokenizer.encode(text))
    # Discard the remainder so all chunks are exactly block_size
    total = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total]
    chunks = [all_ids[i : i + block_size] for i in range(0, total, block_size)]
    return {"input_ids": chunks}


def build_dataset(cfg: dict, tokenizer):
    """Download and preprocess a HuggingFace text dataset."""
    raw = load_dataset(cfg["dataset_name"], cfg["dataset_config"])
    train_raw = raw["train"]

    tokenised = train_raw.map(
        lambda ex: tokenize_and_chunk(ex, tokenizer, cfg["block_size"]),
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenising dataset",
    )
    tokenised.set_format(type="torch", columns=["input_ids"])
    return tokenised


def causal_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Shift input_ids right to build labels for CLM."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    rank, local_rank, world_size = setup_distributed()
    main = is_main_process(rank)

    torch.manual_seed(cfg["seed"] + rank)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO if main else logging.WARNING,
    )

    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available — training on CPU (slow for large models)")

    # ── Mixed precision ──────────────────────────────────────────────────────
    amp_dtype = None
    if cfg["bf16"] and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif cfg["fp16"]:
        amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", cfg["tokenizer_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    logger.info("Preparing dataset: %s/%s", cfg["dataset_name"], cfg["dataset_config"])
    dataset = build_dataset(cfg, tokenizer)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None

    loader = DataLoader(
        dataset,
        batch_size=cfg["per_device_train_batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=causal_collate_fn,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Building model (arch=%s)", cfg["model_arch"])
    model = build_model(cfg).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model parameters: %.1fM", n_params)

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(loader) / cfg["gradient_accumulation_steps"])
    total_steps = (
        cfg["max_steps"] if cfg["max_steps"] > 0
        else steps_per_epoch * cfg["num_train_epochs"]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    writer = None
    if main:
        out = Path(cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(out / "tensorboard"))

        if cfg["use_wandb"]:
            import wandb
            wandb.init(project=cfg["wandb_project"], config=cfg)

    # ── Training ─────────────────────────────────────────────────────────────
    global_step = 0
    best_loss = float("inf")

    for epoch in range(cfg["num_train_epochs"]):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(loader, desc=f"Epoch {epoch+1}", disable=not main)

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(amp_dtype is not None), dtype=amp_dtype or torch.float32):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / cfg["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                running_loss += loss.item() * cfg["gradient_accumulation_steps"]

                # Logging
                if main and global_step % cfg["logging_steps"] == 0:
                    avg_loss = running_loss / cfg["logging_steps"]
                    lr = scheduler.get_last_lr()[0]
                    ppl = math.exp(min(avg_loss, 20))
                    logger.info(
                        "step=%d  loss=%.4f  ppl=%.2f  lr=%.2e",
                        global_step, avg_loss, ppl, lr,
                    )
                    if writer:
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/perplexity", ppl, global_step)
                        writer.add_scalar("train/lr", lr, global_step)
                    if cfg["use_wandb"]:
                        import wandb
                        wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "lr": lr}, step=global_step)
                    running_loss = 0.0

                # Checkpointing
                if main and global_step % cfg["save_steps"] == 0:
                    ckpt_dir = Path(cfg["output_dir"]) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    raw_model = model.module if isinstance(model, DDP) else model
                    raw_model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    logger.info("Saved checkpoint → %s", ckpt_dir)

                if 0 < cfg["max_steps"] <= global_step:
                    break

            progress.set_postfix(loss=f"{loss.item() * cfg['gradient_accumulation_steps']:.4f}")

        if 0 < cfg["max_steps"] <= global_step:
            break

    # ── Final checkpoint ─────────────────────────────────────────────────────
    if main:
        final_dir = Path(cfg["output_dir"]) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("Training complete. Final model saved to %s", final_dir)
        if writer:
            writer.close()

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM pre-training script")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    # Allow any default key to be overridden via CLI
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", action="store_true", default=None)
        elif val is None:
            p.add_argument(f"--{key}", type=str, default=None)
        else:
            p.add_argument(f"--{key}", type=type(val), default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(DEFAULTS)

    # Layer JSON config on top of defaults
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    # Layer CLI overrides on top of everything
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    train(cfg)


if __name__ == "__main__":
    main()
