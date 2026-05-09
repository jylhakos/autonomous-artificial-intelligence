"""
MLflow Tracking Example: Hyperparameter Tuning with Experiment Logging
----------------------------------------------------------------------
Demonstrates MLflow Tracking to log parameters, metrics, and model
artifacts for a scikit-learn classification model across multiple runs.

Run:
    python scripts/tracking/train_and_track.py

Requires (in activated virtual environment):
    pip install mlflow scikit-learn pandas numpy
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "iris-classification-tuning"


def train_model(n_estimators: int, max_depth: int, random_state: int = 42):
    """Train a RandomForest classifier and log results to MLflow."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"rf-n{n_estimators}-d{max_depth}"):
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Train model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Evaluate and log metrics
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log the trained model
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print(
            f"Run: n_estimators={n_estimators}, max_depth={max_depth} | "
            f"Accuracy={accuracy:.4f}, F1={f1:.4f}"
        )


def main():
    # Hyperparameter grid for tuning
    hyperparameter_grid = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 100, "max_depth": None},
    ]

    print(f"Starting hyperparameter tuning. View results at {MLFLOW_TRACKING_URI}")
    for params in hyperparameter_grid:
        train_model(**params)

    print("\nTuning complete. Open the MLflow UI to compare runs:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
