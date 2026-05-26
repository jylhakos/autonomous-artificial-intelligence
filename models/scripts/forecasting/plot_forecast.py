"""
Forecast visualisation: historical usage + LSTM predictions.

Renders a two-panel matplotlib figure:

  Panel 1 (top)    — Full historical daily request counts with a shaded
                     area underneath, plus the future N-day forecast shown
                     as a dashed red line with circle markers.  A vertical
                     grey dotted line marks the boundary between history
                     and forecast.

  Panel 2 (bottom) — Test-set actuals (blue) versus LSTM predictions
                     (green dashed) with error shading (amber fill) and
                     an annotation showing MAE and RMSE in the title.

Usage
-----
    # Activate the forecasting virtual environment first
    cd scripts/forecasting
    source venv/bin/activate

    python plot_forecast.py

    # Save to file and open interactively:
    python plot_forecast.py --days 14 --output figures/forecast.png --show

    # Only save, no window:
    python plot_forecast.py --days 7 --output figures/forecast.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
TARGET_IDX = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def inverse_target(scaler, y_scaled: np.ndarray, n_features: int) -> np.ndarray:
    """Inverse-transform scaled request_count values to original units."""
    dummy = np.zeros((len(y_scaled), n_features), dtype=np.float32)
    dummy[:, TARGET_IDX] = y_scaled.ravel()
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]


def run_forecast(
    model: keras.Model,
    scaler,
    daily: pd.DataFrame,
    lookback: int,
    n_days: int,
) -> tuple[list, list]:
    """
    Run the iterative multi-step forecast from the tail of daily history.

    Returns (future_dates, future_preds) where future_preds are in original
    request-count units.
    """
    n_features = len(FEATURE_COLS)
    scaled_all = scaler.transform(daily[FEATURE_COLS].values.astype(np.float32))
    window = scaled_all[-lookback:].copy()
    last_date = daily.index[-1]

    future_dates, future_preds = [], []

    for step in range(n_days):
        inp = window[np.newaxis, :, :]
        ps = float(model.predict(inp, verbose=0)[0, 0])

        dummy = np.zeros((1, n_features), dtype=np.float32)
        dummy[0, TARGET_IDX] = ps
        real = max(0.0, float(scaler.inverse_transform(dummy)[0, TARGET_IDX]))

        forecast_date = last_date + pd.Timedelta(days=step + 1)
        future_dates.append(forecast_date)
        future_preds.append(real)

        next_row = window[-1].copy()
        next_row[TARGET_IDX] = ps
        dt = forecast_date
        next_row[FEATURE_COLS.index("day_of_week")] = dt.dayofweek / 6.0
        next_row[FEATURE_COLS.index("month")] = (dt.month - 1) / 11.0
        next_row[FEATURE_COLS.index("is_weekend")] = float(dt.dayofweek >= 5)
        window = np.vstack([window[1:], next_row])

    return future_dates, future_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot historical daily usage and LSTM forecast."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory with daily_requests.csv, X_test.npy, etc. (default: data/)",
    )
    parser.add_argument(
        "--model-path",
        default="models/lstm_forecast.keras",
        help="Path to trained Keras model (default: models/lstm_forecast.keras)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of future days to forecast for the top panel (default: 7)",
    )
    parser.add_argument(
        "--output",
        default="figures/forecast.png",
        help="File path to save the figure (default: figures/forecast.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in an interactive window after saving",
    )
    args = parser.parse_args()

    data = Path(args.data_dir)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load prerequisites ------------------------------------------------
    daily_csv = data / "daily_requests.csv"
    if not daily_csv.exists():
        raise FileNotFoundError(
            f"{daily_csv} not found.  Run prepare_data.py first to generate it."
        )

    daily = pd.read_csv(daily_csv, index_col=0, parse_dates=True)

    print(f"Loading model : {args.model_path}")
    model = keras.models.load_model(args.model_path)
    lookback = model.input_shape[1]
    n_features = len(FEATURE_COLS)

    with open(data / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    # ---- Reconstruct test-set predictions (for the bottom panel) ----------
    X_test = np.load(data / "X_test.npy")
    y_test = np.load(data / "y_test.npy")

    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_pred_real = inverse_target(scaler, y_pred_scaled, n_features)
    y_true_real = inverse_target(scaler, y_test, n_features)

    # Align test predictions with their calendar dates
    # The last len(y_test) rows of the daily DataFrame correspond to the test set
    test_dates = daily.index[-len(y_test):]

    mae = float(np.mean(np.abs(y_true_real - y_pred_real)))
    rmse = float(np.sqrt(np.mean((y_true_real - y_pred_real) ** 2)))

    # ---- Run future forecast (for the top panel) --------------------------
    future_dates, future_preds = run_forecast(
        model, scaler, daily, lookback, args.days
    )

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"hspace": 0.45}
    )
    fig.suptitle(
        "Chat Service Daily Load — History and LSTM Forecast",
        fontsize=14,
        fontweight="bold",
    )

    # --- Top panel: full history + future forecast ---
    ax0 = axes[0]
    hist = daily["request_count"]

    ax0.fill_between(hist.index, hist.values, alpha=0.12, color="#3b82f6")
    ax0.plot(
        hist.index,
        hist.values,
        color="#3b82f6",
        linewidth=1.5,
        label="Historical daily requests",
    )
    ax0.plot(
        future_dates,
        future_preds,
        color="#dc2626",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=5,
        label=f"Forecast (+{args.days} days)",
    )
    ax0.axvline(daily.index[-1], color="#6b7280", linestyle=":", linewidth=1)
    ax0.set_title("Historical Request Counts and Future Forecast")
    ax0.set_ylabel("Requests / day")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax0.xaxis.set_major_locator(mdates.AutoDateLocator())

    # --- Bottom panel: test-set actual vs. predicted ---
    ax1 = axes[1]
    ax1.plot(
        test_dates,
        y_true_real,
        color="#3b82f6",
        linewidth=1.5,
        label="Actual (test set)",
    )
    ax1.plot(
        test_dates,
        y_pred_real,
        color="#16a34a",
        linewidth=1.5,
        linestyle="--",
        label="LSTM prediction",
    )
    ax1.fill_between(
        test_dates,
        y_true_real,
        y_pred_real,
        alpha=0.15,
        color="#f59e0b",
        label="Prediction error",
    )
    ax1.set_title(
        f"Test Set: Actual vs. Predicted  |  MAE = {mae:.1f}  ·  RMSE = {rmse:.1f}"
    )
    ax1.set_ylabel("Requests / day")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.autofmt_xdate(rotation=25)

    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    print(f"Figure saved : {out_p}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
