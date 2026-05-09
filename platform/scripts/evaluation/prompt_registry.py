"""
MLflow Prompt Registry: Versioning and Managing Prompts
-------------------------------------------------------
Demonstrates how to register, version, load, and use prompts with
MLflow's Prompt Registry. Prompts are linked to traces and evaluation
results for full lineage tracking.

Run:
    python scripts/evaluation/prompt_registry.py

Requires (in activated virtual environment):
    pip install mlflow[genai]
"""

import mlflow

MLFLOW_TRACKING_URI = "http://localhost:5000"
PROMPT_NAME = "qa-system-prompt"


def register_and_use_prompt():
    """Register a prompt, create a second version, and load both."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Register the initial prompt version
    v1 = mlflow.register_prompt(
        name=PROMPT_NAME,
        template=(
            "You are a helpful assistant. "
            "Answer the following question concisely: {{ question }}"
        ),
        commit_message="Initial prompt: concise answers",
    )
    print(f"Registered prompt '{PROMPT_NAME}' version {v1.version}")

    # Register an improved second version
    v2 = mlflow.register_prompt(
        name=PROMPT_NAME,
        template=(
            "You are an expert assistant. "
            "Provide a clear, accurate, and well-structured answer "
            "to the following question. Cite sources where applicable.\n\n"
            "Question: {{ question }}"
        ),
        commit_message="v2: adds structure and source citation guidance",
    )
    print(f"Registered prompt '{PROMPT_NAME}' version {v2.version}")

    # Load the latest version (production default)
    latest_prompt = mlflow.load_prompt(f"prompts:/{PROMPT_NAME}/latest")
    rendered = latest_prompt.format(question="What is MLflow?")
    print(f"\nLatest prompt rendered:\n{rendered}")

    # Load a specific version for A/B comparison
    v1_prompt = mlflow.load_prompt(f"prompts:/{PROMPT_NAME}/1")
    rendered_v1 = v1_prompt.format(question="What is MLflow?")
    print(f"\nVersion 1 prompt rendered:\n{rendered_v1}")


if __name__ == "__main__":
    register_and_use_prompt()
