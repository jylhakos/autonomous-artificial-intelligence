"""
MLflow GenAI Agent: Tracing a LangChain Agent with an Open-Source LLM
----------------------------------------------------------------------
Creates a LangChain agent powered by a local HuggingFace model, enables
MLflow automatic tracing, and logs the agent interaction to the tracking
server.

Run:
    python scripts/agents/langchain_agent_trace.py

Requires (in activated virtual environment):
    pip install mlflow[genai] langchain langchain-community transformers torch
"""

import mlflow
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "open-source-agent-tracing"


def build_local_llm():
    """Build a local HuggingFace text-generation pipeline."""
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=100,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


def run_agent():
    """Configure MLflow, build the agent, run it, and capture traces."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable automatic tracing for LangChain — one line captures all spans
    mlflow.langchain.autolog()

    llm = build_local_llm()

    # Load a math tool; note: llm-math relies on the chain's LLM
    tools = load_tools(["llm-math"], llm=llm)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    queries = [
        "What is 15 multiplied by 7?",
        "Calculate the square root of 144.",
    ]

    with mlflow.start_run(run_name="langchain-gpt2-agent"):
        for query in queries:
            print(f"\nQuery: {query}")
            try:
                response = agent.invoke({"input": query})
                print(f"Response: {response.get('output', response)}")
            except Exception as exc:
                # Log parse errors so they appear in the trace
                mlflow.log_param("parse_error", str(exc))
                print(f"Agent error (captured in trace): {exc}")

    print(
        f"\nTraces are visible in the MLflow UI at {MLFLOW_TRACKING_URI}"
    )


if __name__ == "__main__":
    run_agent()
