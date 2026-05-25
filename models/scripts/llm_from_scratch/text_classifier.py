"""
Fine-tune the pretrained GPT model as a binary text classifier.

A small classification head is attached to the frozen base model.
Only the classification head is trained; the base model weights are not updated.
This demonstrates the pretrain-then-fine-tune paradigm without any LLM library.

Usage:
    python text_classifier.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from model import GPTModel

# ---------------------------------------------------------------------------
# Small labeled sentiment dataset (for demonstration purposes)
# Replace with a larger dataset file for better results.
# ---------------------------------------------------------------------------
SENTIMENT_DATA: list[tuple[str, int]] = [
    ("This movie is great and I loved it", 1),
    ("Excellent film with outstanding performances", 1),
    ("A wonderful and heartwarming story", 1),
    ("Highly recommend this to everyone", 1),
    ("Brilliant direction and superb acting", 1),
    ("This was a fantastic experience overall", 1),
    ("Amazing cinematography and beautiful scenes", 1),
    ("One of the best films I have ever seen", 1),
    ("A masterpiece of modern cinema", 1),
    ("The storyline was engaging and well written", 1),
    ("Thoroughly enjoyable from start to finish", 1),
    ("An exceptional piece of storytelling", 1),
    ("The performances were moving and memorable", 1),
    ("Stunning visuals and a compelling narrative", 1),
    ("A truly great film that I will watch again", 1),
    ("This movie was terrible and boring", 0),
    ("Waste of time and money", 0),
    ("Poor acting and bad script", 0),
    ("I hated every minute of this film", 0),
    ("Absolutely dreadful and unwatchable", 0),
    ("The worst movie I have seen in years", 0),
    ("Disappointing and completely unoriginal", 0),
    ("Terrible direction and awful dialogue", 0),
    ("A complete failure on every level", 0),
    ("Boring and predictable from start to finish", 0),
    ("The plot made no sense whatsoever", 0),
    ("An absolute waste of everyone involved", 0),
    ("Poorly executed and deeply unenjoyable", 0),
    ("I cannot believe how bad this turned out", 0),
    ("Nothing redeemable about this production", 0),
]

LABEL_NAMES = ["negative", "positive"]
MAX_SEQ_LEN = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """Encodes text samples as fixed-length integer sequences."""

    def __init__(
        self,
        data: list[tuple[str, int]],
        chars: list[str],
        max_len: int = MAX_SEQ_LEN,
    ) -> None:
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.samples: list[tuple[torch.Tensor, int]] = []
        for text, label in data:
            encoded = [self.char_to_idx.get(c, 0) for c in text[:max_len]]
            # Right-pad with zeros to max_len
            encoded += [0] * (max_len - len(encoded))
            self.samples.append((torch.tensor(encoded, dtype=torch.long), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Classifier model
# ---------------------------------------------------------------------------

class TextClassifier(nn.Module):
    """GPT base model with a trainable classification head.

    The base model is kept frozen; only the classification head is updated
    during fine-tuning. The hidden states are mean-pooled over the sequence
    dimension to produce a single vector for classification.
    """

    def __init__(self, base_model: GPTModel, num_classes: int = 2) -> None:
        super().__init__()
        self.base = base_model
        d_model = base_model.token_embedding.embedding_dim

        # Freeze all base model parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # Trainable classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.base.token_embedding(idx) + self.base.position_embedding(positions)
        x = self.base.dropout(x)

        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for block in self.base.blocks:
            x = block(x, mask)
        x = self.base.norm(x)

        # Mean pooling across the sequence dimension
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_classifier() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        chars = checkpoint["tokenizer_chars"]
        print(f"Loaded pretrained model (vocab size: {config['vocab_size']})")
    else:
        print("No pretrained checkpoint found. Run train.py first.")
        print("Using a randomly initialized model for demonstration.")
        chars = sorted(
            set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-")
        )
        config = {
            "vocab_size": len(chars),
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 4,
            "d_ff": 512,
            "max_seq_len": MAX_SEQ_LEN,
            "dropout": 0.0,
        }
        checkpoint = None

    base_model = GPTModel(**config).to(device)
    if checkpoint is not None:
        base_model.load_state_dict(checkpoint["model"])

    model = TextClassifier(base_model, num_classes=2).to(device)

    dataset = SentimentDataset(SENTIMENT_DATA, chars)
    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Training classifier on {train_size} samples, validating on {val_size} samples.")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = torch.tensor(y, dtype=torch.long, device=device) if not isinstance(y, torch.Tensor) else y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = torch.tensor(y, dtype=torch.long, device=device) if not isinstance(y, torch.Tensor) else y.to(device)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()

        accuracy = correct / max(1, val_size)
        if epoch % 5 == 0:
            print(
                f"epoch {epoch:3d}  "
                f"loss {total_loss / len(train_loader):.4f}  "
                f"val acc {accuracy:.2f}"
            )

    # Save the fine-tuned classifier
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(), "chars": chars}, os.path.join(CHECKPOINT_DIR, "classifier.pt"))
    print("\nClassifier saved to checkpoints/classifier.pt")

    # Demonstrate inference
    model.eval()
    print("\nClassifier inference examples:")
    test_inputs = [
        "This is a wonderful and enjoyable experience",
        "This was absolutely terrible and a complete waste",
        "A brilliant film with amazing performances",
        "Dreadful script and poor direction throughout",
        "I really enjoyed this greatly and recommend it",
        "Complete waste of time from start to finish",
    ]
    for text in test_inputs:
        encoded = [chars.index(c) if c in chars else 0 for c in text[:MAX_SEQ_LEN]]
        encoded += [0] * (MAX_SEQ_LEN - len(encoded))
        x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=-1).item()
        print(f"  [{LABEL_NAMES[pred]:8s}]  {text}")


if __name__ == "__main__":
    train_classifier()
