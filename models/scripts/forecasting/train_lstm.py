"""
LSTM model training for daily chat-service load forecasting.

Builds a two-layer LSTM in TensorFlow / Keras, trains it on the sliding-window
arrays produced by prepare_data.py, evaluates on the held-out test set,
and saves the trained model for use by forecast.py and plot_forecast.py.

Model architecture
------------------
    Input  (batch, LOOKBACK, n_features)
      │
      ├── LSTM(units1=64, return_sequences=True)
      ├── Dropout(0.2)
      ├── LSTM(units2=32)
      ├── Dropout(0.2)
      ├── Dense(16, activation="relu")
      └── Dense(1)          ← predicted normalised request_count

How the training works
----------------------
1. Load X_train / y_train arrays written by prepare_data.py.
2. Compile the model with MSE loss and the Adam optimiser.
3. Train with a 10 % validation split drawn from the tail of the training set
   (sequential, no shuffling) so the model never sees future data.
4. EarlyStopping stops training when val_loss stops improving for 10 epochs;
   ReduceLROnPlateau halves the learning rate if val_loss stalls for 5 epochs.
5. After training, evaluate on X_test / y_test, inverse-transform predictions,
   and report MAE and RMSE in original request-count units.
6. Save the best checkpoint to models/lstm_best.keras and the final model to
   models/lstm_forecast.keras.

Evaluation metrics
------------------
    MAE  (Mean Absolute Error)        — average absolute error in requests/day
    RMSE (Root Mean Squared Error)    — penalises large errors more heavily

Usage
-----
    # Activate the forecasting virtual environment first
    cd scripts/forecasting
    source venv/bin/activate

    python train_lstm.py

    # Custom hyperparameters:
    python train_lstm.py --epochs 100 --batch-size 16 --units1 64 --units2 32
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

# Feature list — must match prepare_data.py
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
# Model builder
# ---------------------------------------------------------------------------


def build_model(
    lookback: int,
    n_features: int,
    units1: int = 64,
    units2: int = 32,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Two-layer stacked LSTM with Dropout and a Dense regression head.

    The first LSTM returns the full sequence (return_sequences=True) so the
    second LSTM can see the temporal output of the first.  The second LSTM
    returns only the final hidden state.  Two Dense layers map the state
    to a single predicted value (normalised request_count).
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(lookback, n_features)),
            keras.layers.LSTM(units1, return_sequences=True),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(units2),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ],
        name="lstm_load_forecast",
    )
    return model


# ---------------------------------------------------------------------------
# Inverse-transform helpers
# ---------------------------------------------------------------------------


def inverse_target(
    scaler, y_scaled: np.ndarray, target_idx: int, n_features: int
) -> np.ndarray:
    """
    Inverse-transform only the target column back to original request counts.

    MinMaxScaler expects the full feature matrix.  We construct a dummy
    matrix of zeros, insert the scaled target column, inverse-transform,
    then extract just the target column.
    """
    dummy = np.zeros((len(y_scaled), n_features), dtype=np.float32)
    dummy[:, target_idx] = y_scaled.ravel()
    return scaler.inverse_transform(dummy)[:, target_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an LSTM model to forecast daily chat-service load."
    )
    parser.add_argument("--data-dir",   default="data",
                        help="Directory containing prepared .npy arrays (default: data/)")
    parser.add_argument("--model-dir",  default="models",
                        help="Directory to save trained model files (default: models/)")
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Maximum training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int,   default=16,
                        help="Mini-batch size (default: 16)")
    parser.add_argument("--units1",     type=int,   default=64,
                        help="Units in the first LSTM layer (default: 64)")
    parser.add_argument("--units2",     type=int,   default=32,
                        help="Units in the second LSTM layer (default: 32)")
    parser.add_argument("--dropout",    type=float, default=0.2,
                        help="Dropout rate after each LSTM layer (default: 0.2)")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Initial Adam learning rate (default: 0.001)")
    args = parser.parse_args()

    data = Path(args.data_dir)
    mdir = Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    # ---- Load prepared arrays ----------------------------------------------
    X_train = np.load(data / "X_train.npy")
    y_train = np.load(data / "y_train.npy")
    X_test  = np.load(data / "X_test.npy")
    y_test  = np.load(data / "y_test.npy")

    with open(data / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    lookback, n_features = X_train.shape[1], X_train.shape[2]

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Lookback days    : {lookback}")
    print(f"Feature count    : {n_features}  {FEATURE_COLS}")

    # ---- Build model -------------------------------------------------------
    model = build_model(
        lookback=lookback,
        n_features=n_features,
        units1=args.units1,
        units2=args.units2,
        dropout=args.dropout,
    )
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
        metrics=["mae"],
    )

    # ---- Callbacks ---------------------------------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(mdir / "lstm_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ---- Train -------------------------------------------------------------
    # validation_split takes data from the END of the training set — this
    # is correct for time-series (no future data leaks into training).
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        shuffle=False,    # never shuffle time-series data
        verbose=1,
    )

    # ---- Evaluate on held-out test set -------------------------------------
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_pred = inverse_target(scaler, y_pred_scaled, TARGET_IDX, n_features)
    y_true = inverse_target(scaler, y_test,        TARGET_IDX, n_features)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    print("\n" + "=" * 44)
    print("  Test Evaluation")
    print("=" * 44)
    print(f"  MAE  (requests / day) : {mae:.2f}")
    print(f"  RMSE (requests / day) : {rmse:.2f}")
    print("=" * 44)

    # ---- Save model and training history -----------------------------------
    final_path = mdir / "lstm_forecast.keras"
    model.save(str(final_path))
    print(f"\nFinal model saved : {final_path}")

    history_path = data / "train_history.npz"
    np.savez(
        history_path,
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        mae=history.history["mae"],
        val_mae=history.history["val_mae"],
    )
    print(f"Training history  : {history_path}")


if __name__ == "__main__":
    main()
