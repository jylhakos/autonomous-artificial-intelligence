"""
train_regression.py — Neural Network Regression Training
=========================================================
Trains a configurable MLP regressor on tabular data using PyTorch.

Supported metrics:
  • MSE / RMSE (primary loss)
  • MAE
  • R² score

Quick start (diabetes dataset from scikit-learn):
    python train_regression.py

Custom CSV:
    python train_regression.py --csv_path data.csv --target_column price
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from data_pipeline import DataConfig, build_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    """Fully connected regression network."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: list[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, 1))   # single output — regression
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_preds: list[float] = []
    all_targets: list[float] = []

    for X, y in loader:
        X, y = X.to(device), y.to(device).float()
        preds = model(X)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    mse = mean_squared_error(targets_arr, preds_arr)
    return {
        "mse":  float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae":  float(mean_absolute_error(targets_arr, preds_arr)),
        "r2":   float(r2_score(targets_arr, preds_arr)),
    }


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
    data_cfg = DataConfig(
        dataset_name=cfg.dataset_name,
        csv_path=cfg.csv_path or None,
        target_column=cfg.target_column,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    train_loader, val_loader, test_loader, meta = build_pipeline(data_cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MLPRegressor(
        in_features=meta.num_features,
        hidden_dims=[256, 128, 64],
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d  |  Features: %d", n_params, meta.num_features)

    # ── Loss / optimiser ──────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_rmse = float("inf")

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for X, y in progress:
            X, y = X.to(device), y.to(device).float()
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix(mse=f"{loss.item():.4f}")

        avg_train_mse = running_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion, device)
        lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %d | train_mse=%.4f | val_rmse=%.4f | val_mae=%.4f | val_r2=%.4f | lr=%.2e",
            epoch + 1,
            avg_train_mse,
            val_metrics["rmse"],
            val_metrics["mae"],
            val_metrics["r2"],
            lr,
        )
        writer.add_scalar("train/mse",  avg_train_mse,        epoch)
        writer.add_scalar("val/rmse",   val_metrics["rmse"],  epoch)
        writer.add_scalar("val/mae",    val_metrics["mae"],   epoch)
        writer.add_scalar("val/r2",     val_metrics["r2"],    epoch)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(model.state_dict(), out / "best_model.pt")
            logger.info("  ↑ New best val_rmse=%.4f — checkpoint saved", best_val_rmse)

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(
        "Test | rmse=%.4f | mae=%.4f | r2=%.4f",
        test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"],
    )
    for k, v in test_metrics.items():
        writer.add_scalar(f"test/{k}", v, 0)

    writer.close()
    logger.info("Training complete. Best val_rmse=%.4f", best_val_rmse)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural network regression training")
    p.add_argument("--dataset_name", default="diabetes", help="sklearn regression dataset name")
    p.add_argument("--csv_path", default=None, help="Path to a CSV file")
    p.add_argument("--target_column", default="target")
    p.add_argument("--output_dir", default="./checkpoints/regressor")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
