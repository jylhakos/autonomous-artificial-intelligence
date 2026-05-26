"""
Data preparation pipeline for service load forecasting.

Reads the hourly request log produced by log_analyzer.py (hourly_load.csv),
aggregates it to daily totals, engineers temporal features, normalises the
features with MinMaxScaler, and builds sliding-window (X, y) arrays that
the LSTM trainer expects.

What this script does
---------------------
1. Load hourly_load.csv — each row is one UTC hour of the chat service.
2. Resample to daily totals so the target variable is "requests per day".
3. Add temporal features: day_of_week, month, is_weekend.
4. Normalise all feature columns to [0, 1] with MinMaxScaler.
5. Create sliding windows: for each day d, X[d] = the previous LOOKBACK days
   of scaled features; y[d] = the scaled request_count on day d.
6. Split sequentially (first 80 % → train, last 20 % → test).
7. Save X_train, y_train, X_test, y_test as .npy files and the scaler as a
   pickle so that train_lstm.py and forecast.py can use them.

Outputs (inside --out-dir, default: data/)
------------------------------------------
    daily_requests.csv    human-readable daily aggregation
    X_train.npy           shape (n_train, LOOKBACK, n_features)
    y_train.npy           shape (n_train,)
    X_test.npy            shape (n_test,  LOOKBACK, n_features)
    y_test.npy            shape (n_test,)
    scaler.pkl            fitted MinMaxScaler for inverse-transform

Usage
-----
    # Activate the forecasting virtual environment first
    cd scripts/forecasting
    source venv/bin/activate

    python prepare_data.py

    # With custom paths:
    python prepare_data.py \\
        --csv  ../../llm_from_scratch/logs/hourly_load.csv \\
        --lookback 14 \\
        --out-dir data/
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants — must match across prepare_data.py / train_lstm.py / forecast.py
# ---------------------------------------------------------------------------

LOOKBACK = 7       # default number of past days used as LSTM input
TEST_SPLIT = 0.2   # fraction of days held out for evaluation

FEATURE_COLS = [
    "request_count",
    "avg_latency_ms",
    "avg_completion_tokens",
    "day_of_week",
    "month",
    "is_weekend",
]
TARGET_COL = "request_count"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """
    Load hourly_load.csv and resample to daily totals.

    Columns produced:
        request_count         — sum of hourly request counts
        avg_latency_ms        — mean of hourly average latencies
        avg_completion_tokens — mean of hourly average completion token counts
    """
    df = pd.read_csv(csv_path, parse_dates=["hour"])
    df = df.set_index("hour").sort_index()

    daily = (
        df.resample("D")
        .agg(
            {
                "request_count": "sum",
                "avg_latency_ms": "mean",
                "avg_completion_tokens": "mean",
            }
        )
        .fillna(0)
    )
    return daily


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add day-of-week, month, and is_weekend columns.

    These cyclic features help the model recognise weekly and monthly
    patterns (e.g. lower load on weekends or at the start of the month).
    """
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek        # 0 = Monday … 6 = Sunday
    df["month"] = df.index.month                  # 1 – 12
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def make_windows(
    scaled: np.ndarray,
    lookback: int,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sliding-window pairs from a 2-D scaled array.

    For each position i >= lookback:
        X[i] = scaled[i-lookback : i]    shape (lookback, n_features)
        y[i] = scaled[i, target_idx]     scalar (normalised request_count)
    """
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i])
        y.append(scaled[i, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare daily time-series data for LSTM load forecasting."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(
            "..", "llm_from_scratch", "logs", "hourly_load.csv"
        ),
        help="Path to hourly_load.csv produced by log_analyzer.py",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK,
        help=f"Number of past days used as LSTM input (default: {LOOKBACK})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=TEST_SPLIT,
        help=f"Fraction of data held out for testing (default: {TEST_SPLIT})",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory for prepared arrays (default: data/)",
    )
    args = parser.parse_args()

    # ---- Load and aggregate ------------------------------------------------
    print(f"Loading:  {args.csv}")
    daily = load_and_aggregate(args.csv)
    daily = add_temporal_features(daily)

    first = daily.index[0].date()
    last = daily.index[-1].date()
    print(f"Daily rows: {len(daily)}  |  date range: {first} → {last}")

    min_rows = args.lookback + 5
    if len(daily) < min_rows:
        print(
            f"Error: need at least {min_rows} daily rows to build windows "
            f"(got {len(daily)}). Collect more request data first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Save human-readable daily CSV ------------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    daily_csv = out / "daily_requests.csv"
    daily[FEATURE_COLS].to_csv(daily_csv)
    print(f"Daily CSV saved:  {daily_csv}")

    # ---- Normalise ---------------------------------------------------------
    feature_data = daily[FEATURE_COLS].values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(feature_data)

    scaler_path = out / "scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"Scaler saved:     {scaler_path}")

    # ---- Sliding windows ---------------------------------------------------
    target_idx = FEATURE_COLS.index(TARGET_COL)
    X, y = make_windows(scaled, args.lookback, target_idx)

    # Sequential split — never shuffle time-series data
    split = int(len(X) * (1 - args.test_split))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    np.save(out / "X_train.npy", X_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_test.npy", y_test)

    print(f"\nArrays saved to {out}/")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")
    print(f"  Lookback: {args.lookback} days  |  Features: {FEATURE_COLS}")


if __name__ == "__main__":
    main()
