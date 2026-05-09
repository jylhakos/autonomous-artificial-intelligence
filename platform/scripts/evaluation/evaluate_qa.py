"""
MLflow GenAI Evaluation: Correctness and Relevance Scoring
-----------------------------------------------------------
Evaluates a simple question-answering function using MLflow's built-in
GenAI scorers (Correctness, RelevanceToQuery). Results are logged to
the MLflow tracking server with a full evaluation run.

Run:
    python scripts/evaluation/evaluate_qa.py

Requires (in activated virtual environment):
    pip install mlflow[genai] openai
    export OPENAI_API_KEY=<your-key>   # required for LLM judges
"""

import pandas as pd
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "qa-agent-evaluation"


def simple_qa_agent(inputs: dict) -> str:
    """
    A minimal QA function for demonstration.
    In a real scenario this would call an LLM or RAG pipeline.
    """
    question = inputs.get("question", "").lower()

    answer_map = {
        "capital of france": "The capital of France is Paris.",
        "10 plus 5": "10 plus 5 equals 15.",
        "square root of 144": "The square root of 144 is 12.",
        "largest planet": "The largest planet in our solar system is Jupiter.",
    }

    for key, answer in answer_map.items():
        if key in question:
            return answer

    return "I do not have enough information to answer that question."


def build_evaluation_dataset() -> pd.DataFrame:
    """Create a structured evaluation dataset with inputs and expected outputs."""
    records = [
        {
            "inputs": {"question": "What is the capital of France?"},
            "expectations": {
                "expected_response": "The capital of France is Paris."
            },
        },
        {
            "inputs": {"question": "What is 10 plus 5?"},
            "expectations": {"expected_response": "15"},
        },
        {
            "inputs": {"question": "What is the square root of 144?"},
            "expectations": {"expected_response": "12"},
        },
        {
            "inputs": {"question": "What is the largest planet in our solar system?"},
            "expectations": {"expected_response": "Jupiter"},
        },
    ]
    return pd.DataFrame(records)


def run_evaluation():
    """Run the full MLflow GenAI evaluation pipeline."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    eval_df = build_evaluation_dataset()

    print("Running MLflow GenAI evaluation with Correctness and RelevanceToQuery scorers...")
    print(f"Evaluation dataset has {len(eval_df)} records.\n")

    results = mlflow.genai.evaluate(
        data=eval_df,
        predict_fn=simple_qa_agent,
        scorers=[
            Correctness(),
            RelevanceToQuery(),
        ],
    )

    print("\nEvaluation metrics:")
    print(results.metrics)

    print(f"\nDetailed results are available in the MLflow UI at {MLFLOW_TRACKING_URI}")
    return results


if __name__ == "__main__":
    run_evaluation()
