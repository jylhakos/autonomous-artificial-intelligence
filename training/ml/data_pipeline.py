"""
data_pipeline.py — ML Data Pipeline Utilities
==============================================
Covers the standard data pipeline stages:
  1. Loading  — CSV / HuggingFace datasets / scikit-learn toy datasets
  2. Cleaning — missing values, outliers, type coercion
  3. Feature engineering — encoding, scaling, polynomial features
  4. Splitting — stratified train / validation / test splits
  5. PyTorch Dataset/DataLoader wrapping

Usage:
    from data_pipeline import build_pipeline

    train_loader, val_loader, test_loader, meta = build_pipeline(cfg)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Source — choose one
    dataset_name: str = "iris"          # scikit-learn toy or HuggingFace dataset id
    csv_path: Optional[str] = None      # local CSV file (overrides dataset_name)
    target_column: str = "target"       # column name for the label (CSV only)

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Preprocessing
    scale_features: bool = True
    handle_missing: str = "mean"        # "mean" | "median" | "drop"

    # DataLoader
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42


# ---------------------------------------------------------------------------
# Metadata returned by build_pipeline
# ---------------------------------------------------------------------------

@dataclass
class DataMeta:
    num_features: int = 0
    num_classes: int = 0                # 0 for regression tasks
    class_names: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    task: str = "classification"        # "classification" | "regression"
    scaler: Optional[StandardScaler] = None
    label_encoder: Optional[LabelEncoder] = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_sklearn_dataset(name: str) -> tuple[np.ndarray, np.ndarray, DataMeta]:
    from sklearn import datasets as skds

    loaders = {
        "iris": skds.load_iris,
        "wine": skds.load_wine,
        "breast_cancer": skds.load_breast_cancer,
        "digits": skds.load_digits,
        "diabetes": skds.load_diabetes,   # regression
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown scikit-learn dataset '{name}'. "
            f"Choose from: {list(loaders)}"
        )

    data = loaders[name]()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64 if hasattr(data, "target_names") else np.float32)

    meta = DataMeta(
        num_features=X.shape[1],
        num_classes=len(np.unique(y)) if name != "diabetes" else 0,
        class_names=list(getattr(data, "target_names", [])),
        feature_names=list(getattr(data, "feature_names", [f"x{i}" for i in range(X.shape[1])])),
        task="regression" if name == "diabetes" else "classification",
    )
    logger.info("Loaded sklearn dataset '%s': %s samples, %d features", name, len(X), X.shape[1])
    return X, y, meta


def _load_csv(path: str, target_col: str) -> tuple[np.ndarray, np.ndarray, DataMeta]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}. Columns: {list(df.columns)}")

    y_raw = df[target_col].values
    X_df = df.drop(columns=[target_col])

    # Encode categorical features
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        enc = LabelEncoder()
        X_df[col] = enc.fit_transform(X_df[col].astype(str))

    X = X_df.values.astype(np.float32)

    # Encode labels
    le = None
    if y_raw.dtype == object or str(y_raw.dtype).startswith("U"):
        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(np.int64)
        class_names = list(le.classes_)
        task = "classification"
    elif len(np.unique(y_raw)) <= 50:
        y = y_raw.astype(np.int64)
        class_names = [str(c) for c in np.unique(y_raw)]
        task = "classification"
    else:
        y = y_raw.astype(np.float32)
        class_names = []
        task = "regression"

    meta = DataMeta(
        num_features=X.shape[1],
        num_classes=len(class_names),
        class_names=class_names,
        feature_names=list(X_df.columns),
        task=task,
        label_encoder=le,
    )
    logger.info("Loaded CSV '%s': %d samples, %d features, task=%s", path, len(X), X.shape[1], task)
    return X, y, meta


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """Replace NaN values according to strategy."""
    if not np.any(np.isnan(X)):
        return X
    nan_count = int(np.sum(np.isnan(X)))
    logger.info("Filling %d NaN values with strategy='%s'", nan_count, strategy)
    if strategy == "drop":
        mask = ~np.any(np.isnan(X), axis=1)
        return X[mask]
    fill = np.nanmean(X, axis=0) if strategy == "mean" else np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(fill, idx[1])
    return X


# ---------------------------------------------------------------------------
# Splitting & scaling
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DataConfig,
) -> tuple[np.ndarray, ...]:
    """Return (X_train, X_val, X_test, y_train, y_val, y_test)."""
    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    stratify = y if len(np.unique(y)) < len(y) // 2 else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=cfg.test_ratio,
        random_state=cfg.seed,
        stratify=stratify,
    )
    val_ratio_adjusted = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    stratify_tv = y_trainval if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        random_state=cfg.seed,
        stratify=stratify_tv,
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on train set and apply to all splits."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32), scaler


# ---------------------------------------------------------------------------
# PyTorch wrapping
# ---------------------------------------------------------------------------

def arrays_to_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_pipeline(
    cfg: DataConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, DataMeta]:
    """Execute the full data pipeline and return ready-to-use DataLoaders."""
    # 1. Load
    if cfg.csv_path:
        X, y, meta = _load_csv(cfg.csv_path, cfg.target_column)
    else:
        X, y, meta = _load_sklearn_dataset(cfg.dataset_name)

    # 2. Clean
    X = clean(X, strategy=cfg.handle_missing)

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)

    # 4. Scale
    if cfg.scale_features:
        X_train, X_val, X_test, meta.scaler = scale_features(X_train, X_val, X_test)

    # 5. DataLoaders
    train_loader = arrays_to_loader(X_train, y_train, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    val_loader   = arrays_to_loader(X_val,   y_val,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader  = arrays_to_loader(X_test,  y_test,  cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader, meta


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = DataConfig(dataset_name="iris", batch_size=16)
    train_loader, val_loader, test_loader, meta = build_pipeline(cfg)
    print(f"Task            : {meta.task}")
    print(f"Features        : {meta.num_features}")
    print(f"Classes         : {meta.num_classes}  ({meta.class_names})")
    print(f"Train batches   : {len(train_loader)}")
    print(f"Val   batches   : {len(val_loader)}")
    print(f"Test  batches   : {len(test_loader)}")
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shape     : X={tuple(X_batch.shape)}  y={tuple(y_batch.shape)}")
