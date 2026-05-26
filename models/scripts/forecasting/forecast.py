"""
Daily chat-service load forecast using the trained LSTM model.

Loads the trained model (models/lstm_forecast.keras) and the most recent
daily request data, then predicts the next N days' request counts using an
iterative multi-step approach: predict one day, roll the input window one
step forward, repeat.

How multi-step forecasting works
---------------------------------
1. The last LOOKBACK rows of the daily history are used as the initial
   input window, shape (1, LOOKBACK, n_features).
2. The model predicts the normalised request_count for day D+1.
3. The predicted value is substituted into a new row (temporal features
   are derived from the calendar date of day D+1).
4. The window is shifted forward by one day (oldest row dropped, new row
   appended) to create the input for day D+2.
5. This repeats for the requested number of forecast days.

Multi-step forecasting accumulates error with each step because the
predicted values are fed back as inputs.  Short horizons (1–7 days)
are typically more reliable than longer ones.

Outputs
-------
    data/forecast.csv     date, predicted_requests (printed and saved)

Usage
-----
    # Activate the forecasting virtual environment first
    cd scripts/forecasting
    source venv/bin/activate

    python forecast.py

    # Forecast the next 7 days:
    python forecast.py --days 7

    # Use a different source CSV:
    python forecast.py --days 3 \\
        --csv ../../llm_from_scratch/logs/hourly_load.csv
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Feature list — must match prepare_data.py exactly (order matters)
FEATURE_COLS = [
    "request_count",
    "avg_latency_ms",
    "avg_completion_tokens",
    "day_of_week",
    "month",
    "is_weekend",
]
TARGET_IDX = 0  # request_count is the first column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_hourly_as_daily(csv_path: str) -> pd.DataFrame:
    """
    Load hourly_load.csv and resample to daily totals.

    Falls back to this function when data/daily_requests.csv does not yet
    exist (i.e. prepare_data.py has not been run with this CSV).
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
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["is_weekend"] = (daily.index.dayofweek >= 5).astype(int)
    return daily[FEATURE_COLS]


def inverse_target(
    scaler, pred_scaled: float, n_features: int
) -> float:
    """Inverse-transform a single scaled request_count to original units."""
    dummy = np.zeros((1, n_features), dtype=np.float32)
    dummy[0, TARGET_IDX] = pred_scaled
    return float(scaler.inverse_transform(dummy)[0, TARGET_IDX])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast next day(s) of chat-service request load."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(
            "..", "llm_from_scratch", "logs", "hourly_load.csv"
        ),
        help="Path to hourly_load.csv (used as fallback if data/daily_requests.csv "
             "is not found)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of future days to forecast (default: 1)",
    )
    parser.add_argument(
        "--model-path",
        default="models/lstm_forecast.keras",
        help="Path to the trained Keras model (default: models/lstm_forecast.keras)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory that contains scaler.pkl and daily_requests.csv (default: data/)",
    )
    args = parser.parse_args()

    data = Path(args.data_dir)

    # ---- Load model and scaler ---------------------------------------------
    print(f"Loading model  : {args.model_path}")
    model = keras.models.load_model(args.model_path)
    lookback = model.input_shape[1]

    with open(data / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    n_features = len(FEATURE_COLS)

    # ---- Prepare the most recent lookback window ---------------------------
    daily_csv = data / "daily_requests.csv"
    if daily_csv.exists():
        daily = pd.read_csv(daily_csv, index_col=0, parse_dates=True)[FEATURE_COLS]
        print(f"Using prepared daily data : {daily_csv}")
    else:
        print(f"daily_requests.csv not found — loading from {args.csv}")
        daily = load_hourly_as_daily(args.csv)

    if len(daily) < lookback:
        raise ValueError(
            f"Need at least {lookback} days of history to forecast; "
            f"only {len(daily)} rows available.  Run prepare_data.py first."
        )

    scaled_all = scaler.transform(daily.values.astype(np.float32))
    window = scaled_all[-lookback:].copy()   # shape (lookback, n_features)

    last_date = daily.index[-1]

    # ---- Iterative multi-step forecast ------------------------------------
    predictions = []

    for step in range(args.days):
        inp = window[np.newaxis, :, :]                  # (1, lookback, n_features)
        pred_scaled = float(model.predict(inp, verbose=0)[0, 0])
        pred_real = max(0, round(inverse_target(scaler, pred_scaled, n_features)))

        forecast_date = last_date + pd.Timedelta(days=step + 1)
        predictions.append((forecast_date.date(), int(pred_real)))

        # Build the next row: copy the last known row, overwrite dynamic cols
        next_row = window[-1].copy()
        next_row[TARGET_IDX] = pred_scaled

        # Temporal features are deterministic — normalise to [0, 1] manually
        dt = forecast_date
        dow_idx = FEATURE_COLS.index("day_of_week")
        mon_idx = FEATURE_COLS.index("month")
        wkd_idx = FEATURE_COLS.index("is_weekend")
        next_row[dow_idx] = dt.dayofweek / 6.0
        next_row[mon_idx] = (dt.month - 1) / 11.0
        next_row[wkd_idx] = float(dt.dayofweek >= 5)

        window = np.vstack([window[1:], next_row])

    # ---- Print and save results --------------------------------------------
    print("\n" + "=" * 44)
    print("  Chat Service Load Forecast")
    print("=" * 44)
    for date, count in predictions:
        day_name = pd.Timestamp(date).day_name()
        print(f"  {date}  ({day_name:<9})  →  {count:>7} requests / day")
    print("=" * 44)

    out_path = data / "forecast.csv"
    pd.DataFrame(predictions, columns=["date", "predicted_requests"]).to_csv(
        out_path, index=False
    )
    print(f"\nForecast saved : {out_path}")


if __name__ == "__main__":
    main()
