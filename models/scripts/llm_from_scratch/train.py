"""
Pre-train a GPT-style language model from scratch.

The model is trained with next-token prediction on a character-level corpus.
No existing LLM library or pretrained weights are used.

Usage:
    python train.py
    python train.py --data data/train.txt --iters 3000 --d-model 64
"""

import argparse
import os

import torch
import torch.optim as optim

from model import GPTModel
from tokenizer import CharTokenizer

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
BLOCK_SIZE = 128       # Context window length (tokens)
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "data/train.txt"

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog.\n"
    "Natural language processing is a branch of artificial intelligence.\n"
    "Large language models learn statistical patterns from text data.\n"
    "Transformers use self-attention mechanisms to process sequences.\n"
    "The model predicts the next token given all previous tokens.\n"
    "Training involves minimizing cross-entropy loss over many iterations.\n"
    "Fine-tuning adapts a pretrained model to a specific downstream task.\n"
    "Tokenization converts raw text into sequences of integers.\n"
    "Embeddings map token identifiers to continuous vector representations.\n"
    "Positional encodings help the model understand token order in a sequence.\n"
    "The attention mechanism allows the model to focus on relevant context.\n"
    "Residual connections help gradient flow through deep neural networks.\n"
    "Layer normalization stabilizes training of deep transformer models.\n"
    "The feed-forward network adds non-linear capacity to each transformer block.\n"
    "Weight tying shares parameters between the embedding and output layers.\n"
) * 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (inputs, targets) from data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """Estimate average loss on train and validation splits."""
    model.eval()
    losses: dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Prepare training data
    if not os.path.exists(args.data):
        os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
        with open(args.data, "w", encoding="utf-8") as fh:
            fh.write(SAMPLE_TEXT)
        print(f"Created sample training data at '{args.data}'")

    with open(args.data, "r", encoding="utf-8") as fh:
        text = fh.read()
    print(f"Dataset: {len(text):,} characters")

    # Tokenize
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    # Build model
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.block_size,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    for iteration in range(args.iters):
        # Periodic evaluation
        if iteration % args.eval_interval == 0 or iteration == args.iters - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                args.block_size, args.batch_size, args.eval_iters, device,
            )
            print(
                f"step {iteration:5d}  "
                f"train loss {losses['train']:.4f}  "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "tokenizer_chars": tokenizer.chars,
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "d_ff": args.d_ff,
                        "max_seq_len": args.block_size,
                        "dropout": 0.0,  # disable dropout at inference
                    },
                }
                save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(checkpoint, save_path)
                print(f"  checkpoint saved (val loss {best_val_loss:.4f})")

        # Training step
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    print("\nTraining complete.")

    # Generate a short sample to show the model learned something
    model.eval()
    prompt = "The model"
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print("\nSample output:")
    print(tokenizer.decode(generated[0].tolist()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a small GPT model from scratch.")
    parser.add_argument("--data", default=DATA_PATH, help="Path to plain-text training file")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--iters", type=int, default=MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=EVAL_ITERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--d-ff", type=int, default=D_FF)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
