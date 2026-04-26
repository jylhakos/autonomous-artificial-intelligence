"""Tests for pydantic_ai_example.py using pytest.

These tests use mocking to avoid requiring a live Ollama server,
making them safe to run in any environment with the dependencies installed.

Usage (with virtual environment activated):
    source venv/bin/activate
    pytest tests/test_pydantic_ai.py -v
"""

from unittest.mock import MagicMock

import pytest


class TestSafeResponse:
    """Tests for the SafeResponse Pydantic model."""

    def test_safe_response_accepts_valid_data(self):
        """Verify that SafeResponse accepts well-formed input."""
        from pydantic_ai_example import SafeResponse

        response = SafeResponse(
            response="Paris is the capital of France.",
            is_safe=True,
            topic="Geography",
        )
        assert response.response == "Paris is the capital of France."
        assert response.is_safe is True
        assert response.topic == "Geography"

    def test_safe_response_can_be_marked_unsafe(self):
        """Verify that is_safe can be set to False for inappropriate content."""
        from pydantic_ai_example import SafeResponse

        response = SafeResponse(
            response="Some inappropriate content.",
            is_safe=False,
            topic="Inappropriate",
        )
        assert response.is_safe is False

    def test_safe_response_model_dump_returns_dict(self):
        """Verify that model_dump() serializes the model to a plain dict."""
        from pydantic_ai_example import SafeResponse

        response = SafeResponse(
            response="Hello world.",
            is_safe=True,
            topic="Greeting",
        )
        data = response.model_dump()
        assert isinstance(data, dict)

    def test_safe_response_model_dump_has_all_fields(self):
        """Verify that model_dump() includes all expected fields."""
        from pydantic_ai_example import SafeResponse

        response = SafeResponse(
            response="Test.",
            is_safe=True,
            topic="Test",
        )
        data = response.model_dump()
        assert set(data.keys()) == {"response", "is_safe", "topic"}

    def test_safe_response_requires_all_fields(self):
        """Verify that SafeResponse raises a validation error on missing fields."""
        from pydantic import ValidationError

        from pydantic_ai_example import SafeResponse

        with pytest.raises(ValidationError):
            SafeResponse(response="Missing fields")


class TestCreateOllamaModel:
    """Tests for the create_ollama_model factory function."""

    def test_create_ollama_model_returns_openai_chat_model(self):
        """Verify that create_ollama_model returns an OpenAIChatModel instance."""
        from pydantic_ai.models.openai import OpenAIChatModel

        from pydantic_ai_example import create_ollama_model

        model = create_ollama_model()
        assert isinstance(model, OpenAIChatModel)

    def test_create_ollama_model_is_not_none(self):
        """Verify that create_ollama_model never returns None."""
        from pydantic_ai_example import create_ollama_model

        model = create_ollama_model()
        assert model is not None


class TestRunSafeQuery:
    """Tests for the run_safe_query function."""

    def test_run_safe_query_success_structure(self):
        """Verify that a successful query returns the expected dict shape."""
        from pydantic_ai_example import SafeResponse, run_safe_query

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = SafeResponse(
            response="Water is H2O.",
            is_safe=True,
            topic="Chemistry",
        )
        mock_agent.run_sync.return_value = mock_result

        result = run_safe_query(mock_agent, "What is water?")

        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["response"] == "Water is H2O."
        assert result["data"]["is_safe"] is True
        assert result["data"]["topic"] == "Chemistry"

    def test_run_safe_query_handles_connection_error(self):
        """Verify that connection errors are caught and returned as failures."""
        from pydantic_ai_example import run_safe_query

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = Exception("Connection refused")

        result = run_safe_query(mock_agent, "What is water?")

        assert result["success"] is False
        assert result["data"] is None
        assert "Connection refused" in result["error"]

    def test_run_safe_query_always_has_required_keys(self):
        """Verify that all required keys are present regardless of outcome."""
        from pydantic_ai_example import run_safe_query

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = RuntimeError("Model error")

        result = run_safe_query(mock_agent, "Test query")

        assert "success" in result
        assert "data" in result
        assert "error" in result

    def test_run_safe_query_calls_agent_run_sync(self):
        """Verify that run_safe_query calls agent.run_sync with the query."""
        from pydantic_ai_example import SafeResponse, run_safe_query

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = SafeResponse(
            response="42.", is_safe=True, topic="Math"
        )
        mock_agent.run_sync.return_value = mock_result

        run_safe_query(mock_agent, "What is 6 times 7?")

        mock_agent.run_sync.assert_called_once_with("What is 6 times 7?")
