"""Guardrails AI Example with Ollama via LiteLLM.

This script demonstrates how to use the Guardrails AI framework to validate
LLM outputs for profanity-free content, connecting to a local Ollama server
using LiteLLM as a bridge.

Guardrails AI runs Input/Output Guards in your application that detect, quantify
and mitigate the presence of specific types of risks. It also helps generate
structured data from LLMs.

References:
    - https://github.com/guardrails-ai/guardrails
    - https://guardrailsai.com/guardrails/docs

Prerequisites:
    - Ollama running locally: https://ollama.ai
    - LLM pulled: ollama pull llama3
    - Virtual environment activated: source venv/bin/activate
    - Dependencies installed: pip install -r requirements-guardrails.txt
    - Hub validator installed: guardrails hub install hub://guardrails/profanity_free

Usage:
    source venv/bin/activate
    python guardrails_ai_example.py
"""

import litellm
from guardrails import Guard
from guardrails.hub import ProfanityFree

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "ollama/llama3"


def create_guard() -> Guard:
    """Create a Guard instance with ProfanityFree validator.

    Returns:
        A Guard object configured with the ProfanityFree validator.
    """
    return Guard().use(ProfanityFree())


def validate_response(guard: Guard, user_message: str) -> dict:
    """Send a message to Ollama through the Guard validator.

    The Guard intercepts the LLM call and validates the output against
    the configured validators before returning the result.

    Args:
        guard: The Guardrails Guard instance with validators configured.
        user_message: The prompt to send to the LLM.

    Returns:
        dict with keys:
            - 'success' (bool): Whether validation passed.
            - 'output' (str | None): The validated LLM response, or None on failure.
            - 'error' (str | None): The error message if validation failed.
    """
    try:
        validated_response = guard(
            litellm.completion,
            model=OLLAMA_MODEL,
            api_base=OLLAMA_BASE_URL,
            messages=[{"role": "user", "content": user_message}],
        )
        return {
            "success": True,
            "output": validated_response.validated_output,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "output": None,
            "error": str(exc),
        }


def main() -> None:
    """Run Guardrails AI validation examples against a local Ollama LLM."""
    guard = create_guard()

    test_cases = [
        "Write a polite and professional greeting.",
        "Say hello in a friendly way.",
        "What are some tips for good communication?",
    ]

    print("=== Guardrails AI: ProfanityFree Validation with Ollama ===\n")

    for message in test_cases:
        print(f"Prompt: {message}")
        result = validate_response(guard, message)
        if result["success"]:
            print(f"  Validated output: {result['output']}")
        else:
            print(f"  Guardrail blocked or error: {result['error']}")
        print()


if __name__ == "__main__":
    main()
