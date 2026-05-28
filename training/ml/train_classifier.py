"""
train_classifier.py — Neural Network Classifier Training
=========================================================
Trains a configurable MLP or CNN classifier on tabular data (or image data)
using PyTorch.  Plugs directly into the data_pipeline module.

Supported tasks:
  • Binary and multi-class classification on tabular data
  • MNIST / CIFAR-style image classification (via torchvision)

Quick start (tabular — iris dataset):
    python train_classifier.py

Tabular with custom CSV:
    python train_classifier.py --csv_path data.csv --target_column label

Image classification (MNIST):
    python train_classifier.py --task image --dataset_name MNIST

"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets as tvdatasets, transforms

from sklearn.metrics import classification_report
from tqdm import tqdm

from data_pipeline import DataConfig, DataMeta, build_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-layer perceptron for tabular classification."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: list[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """Small CNN for 1-channel 28×28 images (MNIST-style)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 14×14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 7×7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Image data loaders
# ---------------------------------------------------------------------------

def build_image_loaders(dataset_name: str, batch_size: int):
    """Return (train_loader, val_loader, test_loader, num_classes) for MNIST."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    root = "./data"

    cls = getattr(tvdatasets, dataset_name, None)
    if cls is None:
        raise ValueError(f"torchvision dataset '{dataset_name}' not found.")

    train_ds = cls(root, train=True,  download=True, transform=tf)
    test_ds  = cls(root, train=False, download=True, transform=tf)

    # 90 / 10 train-val split
    n_val = int(0.1 * len(train_ds))
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [len(train_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    kwargs = dict(batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
    return (
        torch.utils.data.DataLoader(train_ds, shuffle=True,  **kwargs),
        torch.utils.data.DataLoader(val_ds,   shuffle=False, **kwargs),
        torch.utils.data.DataLoader(test_ds,  shuffle=False, **kwargs),
        len(train_ds.dataset.classes),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: argparse.Namespace) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out / "tensorboard"))

    # ── Data ─────────────────────────────────────────────────────────────────
    if cfg.task == "image":
        train_loader, val_loader, test_loader, num_classes = build_image_loaders(
            cfg.dataset_name, cfg.batch_size
        )
        model: nn.Module = SimpleCNN(num_classes=num_classes).to(device)
    else:
        data_cfg = DataConfig(
            dataset_name=cfg.dataset_name,
            csv_path=cfg.csv_path or None,
            target_column=cfg.target_column,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )
        train_loader, val_loader, test_loader, meta = build_pipeline(data_cfg)
        model = MLP(
            in_features=meta.num_features,
            num_classes=meta.num_classes,
            hidden_dims=[256, 128, 64],
            dropout=cfg.dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    # ── Loss / optimiser ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = math.ceil(len(train_loader) / 1) * cfg.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.learning_rate, total_steps=total_steps
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for X, y in progress:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e",
            epoch + 1, avg_train_loss, val_loss, val_acc, lr,
        )
        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out / "best_model.pt")
            logger.info("  ↑ New best val_acc=%.4f — checkpoint saved", best_val_acc)

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("Test loss=%.4f  test_acc=%.4f", test_loss, test_acc)
    writer.add_scalar("test/loss", test_loss, 0)
    writer.add_scalar("test/accuracy", test_acc, 0)

    # Detailed classification report (tabular only)
    if cfg.task == "tabular":
        all_preds, all_targets = [], []
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                preds = model(X.to(device)).argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_targets.extend(y.tolist())
        print("\nClassification Report:\n")
        print(classification_report(all_targets, all_preds))

    writer.close()
    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural network classifier training")
    p.add_argument("--task", choices=["tabular", "image"], default="tabular")
    p.add_argument("--dataset_name", default="iris", help="sklearn dataset name or torchvision class")
    p.add_argument("--csv_path", default=None, help="Path to a CSV file (tabular task)")
    p.add_argument("--target_column", default="target")
    p.add_argument("--output_dir", default="./checkpoints/classifier")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
