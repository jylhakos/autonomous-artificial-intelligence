"""
train_rnn_ddp.py — Multi-GPU RNN Pre-training with PyTorch DDP
===============================================================
Pre-trains an LSTM or GRU on synthetic multivariate time-series windows using
native PyTorch Distributed Data Parallel (DDP) on Linux.

Quick start (2 GPUs):
    torchrun --standalone --nproc_per_node=2 train_rnn_ddp.py

Switch recurrent cell type:
    torchrun --standalone --nproc_per_node=4 train_rnn_ddp.py --cell_type gru
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class SyntheticTimeSeriesDataset(Dataset):
    """Create sliding windows over synthetic multivariate signals."""

    def __init__(
        self,
        num_samples: int,
        window_size: int,
        input_dim: int,
        seed: int,
    ) -> None:
        if num_samples <= window_size:
            raise ValueError("num_samples must be greater than window_size")

        generator = torch.Generator().manual_seed(seed)
        timeline = torch.linspace(0, 8 * math.pi, steps=num_samples, dtype=torch.float32)

        features: list[torch.Tensor] = []
        for feature_idx in range(input_dim):
            frequency = 0.25 + (feature_idx * 0.08)
            phase = feature_idx * 0.35
            waveform = (
                torch.sin(timeline * frequency + phase)
                + 0.5 * torch.cos(timeline * (frequency * 0.5) + phase)
            )
            noise = 0.05 * torch.randn(num_samples, generator=generator)
            features.append(waveform + noise)

        self.series = torch.stack(features, dim=1)
        self.window_size = window_size

    def __len__(self) -> int:
        return self.series.size(0) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size]
        return x, y


class ForecastRNN(nn.Module):
    """Sequence model for next-step forecasting with LSTM or GRU cells."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        cell_type: str,
    ) -> None:
        super().__init__()

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        return self.projection(outputs[:, -1, :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU RNN pre-training with PyTorch DDP")
    parser.add_argument("--cell_type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--window_size", type=int, default=168)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", default="./checkpoints/rnn_ddp")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, global_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_dataloader(cfg: argparse.Namespace) -> tuple[DataLoader, DistributedSampler]:
    dataset = SyntheticTimeSeriesDataset(
        num_samples=cfg.num_samples,
        window_size=cfg.window_size,
        input_dim=cfg.input_dim,
        seed=cfg.seed,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return loader, sampler


def train(cfg: argparse.Namespace) -> None:
    local_rank, global_rank, world_size, device = setup_distributed()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    torch.manual_seed(cfg.seed + global_rank)
    torch.backends.cudnn.benchmark = True

    if global_rank == 0:
        logger.info("Initialized DDP training on %d GPU processes", world_size)

    loader, sampler = build_dataloader(cfg)

    model = ForecastRNN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.input_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        cell_type=cfg.cell_type,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    output_dir = Path(cfg.output_dir)
    if global_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = torch.tensor(running_loss / max(len(loader), 1), device=device)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss / world_size

        if global_rank == 0:
            logger.info(
                "rank=0 epoch=%d/%d cell=%s world_size=%d avg_loss=%.6f",
                epoch + 1,
                cfg.epochs,
                cfg.cell_type,
                world_size,
                epoch_loss.item(),
            )

    if global_rank == 0:
        checkpoint_path = output_dir / f"{cfg.cell_type}_forecast_ddp.pt"
        torch.save(model.module.state_dict(), checkpoint_path)
        logger.info("rank=0 saved checkpoint to %s", checkpoint_path)

    cleanup_distributed()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUs are required for train_rnn_ddp.py")

    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()