"""Pydantic AI Example with Ollama: Type-Safe Structured Output.

This script demonstrates how to use Pydantic AI with structured output
validation, connecting to a local Ollama instance via its OpenAI-compatible
API. Pydantic AI guarantees that the LLM response conforms to a strict
Pydantic model schema, providing type-safe agentic guardrails.

Pydantic AI supports Ollama natively through the OpenAI-compatible endpoint
exposed at http://localhost:11434/v1.

References:
    - https://github.com/pydantic/pydantic-ai
    - https://ai.pydantic.dev/

Prerequisites:
    - Ollama running locally: https://ollama.ai
    - LLM pulled: ollama pull llama3
    - Virtual environment activated: source venv/bin/activate
    - Dependencies installed: pip install -r requirements-guardrails.txt

Usage:
    source venv/bin/activate
    python pydantic_ai_example.py
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3"


class SafeResponse(BaseModel):
    """Structured output model that enforces a safe, typed LLM response.

    Pydantic validates the LLM output against this schema. If validation
    fails, the agent automatically re-prompts the LLM to correct its output.
    """

    response: str = Field(description="The LLM response to the user question.")
    is_safe: bool = Field(
        description="True if the content is appropriate for all audiences."
    )
    topic: str = Field(description="The main topic of the response in one or two words.")


def create_ollama_model() -> OpenAIChatModel:
    """Create an OpenAIChatModel configured to target a local Ollama server.

    Ollama exposes an OpenAI-compatible REST API at /v1, so Pydantic AI
    can connect to it using the OpenAIChatModel with a custom base_url.

    Returns:
        An OpenAIChatModel instance pointing to the local Ollama server.
    """
    return OpenAIChatModel(
        model_name=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # Ollama does not require a real API key
    )


def create_agent(model: OpenAIChatModel) -> Agent:
    """Create a Pydantic AI Agent with structured SafeResponse output.

    The agent is configured with an output_type of SafeResponse, which
    forces the LLM to return data conforming to the Pydantic model schema.

    Args:
        model: The OpenAIChatModel instance to use for inference.

    Returns:
        A Pydantic AI Agent configured for safe, structured responses.
    """
    return Agent(
        model=model,
        output_type=SafeResponse,
        instructions=(
            "You are a helpful and safe AI assistant. "
            "Always provide accurate, polite, and appropriate responses. "
            "Set is_safe to True only when the response content is appropriate "
            "for all audiences. Identify the main topic in one or two words."
        ),
    )


def run_safe_query(agent: Agent, query: str) -> dict:
    """Run a query through the Pydantic AI agent with structured output validation.

    The agent sends the query to the LLM and validates that the response
    matches the SafeResponse schema. If it does not, the agent retries.

    Args:
        agent: The Pydantic AI Agent instance.
        query: The user's question or prompt.

    Returns:
        dict with keys:
            - 'success' (bool): Whether the query completed without error.
            - 'data' (dict | None): The validated SafeResponse as a dict.
            - 'error' (str | None): The error message if the query failed.
    """
    try:
        result = agent.run_sync(query)
        return {
            "success": True,
            "data": result.output.model_dump(),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "data": None,
            "error": str(exc),
        }


def main() -> None:
    """Run Pydantic AI structured output guardrail examples with Ollama."""
    model = create_ollama_model()
    agent = create_agent(model)

    queries = [
        "What are the benefits of regular exercise?",
        "Explain the water cycle in simple terms.",
        "What is the capital of France?",
    ]

    print("=== Pydantic AI: Structured Output Guardrails with Ollama ===\n")

    for query in queries:
        print(f"Query: {query}")
        result = run_safe_query(agent, query)
        if result["success"]:
            data = result["data"]
            print(f"  Response : {data['response']}")
            print(f"  Is safe  : {data['is_safe']}")
            print(f"  Topic    : {data['topic']}")
        else:
            print(f"  Error: {result['error']}")
        print()


if __name__ == "__main__":
    main()
