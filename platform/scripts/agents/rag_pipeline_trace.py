"""
MLflow Tracing: Manual Span Instrumentation for a RAG Pipeline
--------------------------------------------------------------
Demonstrates manual MLflow tracing with @mlflow.trace decorators on
a simulated Retrieval-Augmented Generation (RAG) pipeline. Each stage
(retrieval, augmentation, generation) is captured as a structured span.

Run:
    python scripts/agents/rag_pipeline_trace.py

Requires (in activated virtual environment):
    pip install mlflow[genai]
"""

import mlflow
from typing import List

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "rag-pipeline-tracing"

# Simulated document store
DOCUMENT_STORE = {
    "mlflow": (
        "MLflow is an open-source platform for managing the end-to-end machine "
        "learning lifecycle. It includes tracking, projects, models, and a model "
        "registry. MLflow is vendor-neutral and supports AI agents, LLM applications, "
        "and ML models."
    ),
    "tracing": (
        "MLflow Tracing captures the full execution graph of every agent interaction: "
        "every LLM call, tool invocation, and retrieval step with inputs, outputs, "
        "token counts, and latency. It is OpenTelemetry-compatible."
    ),
    "evaluation": (
        "MLflow Evaluation provides a framework to score agent outputs using "
        "deterministic tests, LLM judges, and human feedback. Built-in scorers "
        "include Correctness, Safety, and RelevanceToQuery."
    ),
}


@mlflow.trace(name="retrieve-documents", span_type="RETRIEVER")
def retrieve_documents(query: str) -> List[str]:
    """Simulate document retrieval based on keyword matching."""
    results = []
    query_lower = query.lower()
    for keyword, content in DOCUMENT_STORE.items():
        if keyword in query_lower:
            results.append(content)
    if not results:
        results.append("No directly matching documents found.")
    return results


@mlflow.trace(name="augment-context", span_type="CHAIN")
def augment_context(query: str, documents: List[str]) -> str:
    """Build an augmented prompt from the retrieved documents."""
    context = "\n\n".join(documents)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer based on the context above:"
    )


@mlflow.trace(name="generate-response", span_type="LLM")
def generate_response(augmented_prompt: str) -> str:
    """Simulate LLM generation. Replace with an actual LLM call in production."""
    # In production, call: openai.chat.completions.create(...) or similar
    if "MLflow" in augmented_prompt or "mlflow" in augmented_prompt:
        return (
            "MLflow is an open-source AI platform for managing ML experiments, "
            "agents, and LLM applications with built-in tracing, evaluation, "
            "prompt versioning, and governance capabilities."
        )
    return "Based on the provided context, I cannot find a specific answer."


@mlflow.trace(name="rag-pipeline", span_type="CHAIN")
def rag_pipeline(query: str) -> str:
    """Full RAG pipeline: retrieve, augment, generate."""
    documents = retrieve_documents(query)
    augmented_prompt = augment_context(query, documents)
    response = generate_response(augmented_prompt)
    return response


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    queries = [
        "What is MLflow and what does it support?",
        "How does MLflow tracing work?",
        "How can I evaluate my AI agent with MLflow?",
    ]

    with mlflow.start_run(run_name="rag-manual-tracing"):
        for query in queries:
            print(f"\nQuery: {query}")
            response = rag_pipeline(query)
            print(f"Response: {response}")

    print(f"\nTraces are visible in the MLflow UI at {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
