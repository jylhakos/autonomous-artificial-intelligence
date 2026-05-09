"""
MLflow Model Registry: Log, Register, and Load a Model
-------------------------------------------------------
Demonstrates how to log a trained scikit-learn model to MLflow, register
it in the Model Registry, and load it for inference. This supports the
MLflow Models (Deployment) and Model Registry (Governance) pillars.

Run:
    python scripts/deployment/register_model.py

Requires (in activated virtual environment):
    pip install mlflow scikit-learn
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "model-registry-demo"
MODEL_NAME = "iris-classifier"


def train_and_register():
    """Train a model, log it to MLflow, and register it in the Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="register-iris-rf") as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", 100)

        # Log and register in one call
        model_info = mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        print(f"Model logged. URI: {model_info.model_uri}")
        print(f"Accuracy: {accuracy:.4f}")

    return model_info.model_uri


def load_and_predict(model_uri: str):
    """Load the registered model and run a sample prediction."""
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Sample: predict class for a new iris measurement
    sample_input = [[5.1, 3.5, 1.4, 0.2]]
    prediction = loaded_model.predict(sample_input)
    iris_classes = ["setosa", "versicolor", "virginica"]

    print(f"\nSample prediction for {sample_input}:")
    print(f"  Predicted class index: {prediction[0]}")
    print(f"  Predicted species: {iris_classes[prediction[0]]}")


if __name__ == "__main__":
    model_uri = train_and_register()
    load_and_predict(model_uri)

    print(
        f"\nView the registered model in the MLflow UI at {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}"
    )
