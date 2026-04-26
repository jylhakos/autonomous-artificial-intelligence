"""Tests for guardrails_ai_example.py using pytest.

These tests use mocking to avoid requiring a live Ollama server or
Guardrails AI validators, making them safe to run in any environment.

Usage (with virtual environment activated):
    source venv/bin/activate
    pytest tests/test_guardrails_ai.py -v
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCreateGuard:
    """Tests for the create_guard factory function."""

    def test_create_guard_returns_guard_instance(self):
        """Verify that create_guard returns a Guard object."""
        from guardrails import Guard

        from guardrails_ai_example import create_guard

        guard = create_guard()
        assert isinstance(guard, Guard)

    def test_create_guard_is_not_none(self):
        """Verify that create_guard never returns None."""
        from guardrails_ai_example import create_guard

        guard = create_guard()
        assert guard is not None

    def test_create_guard_returns_new_instance_each_call(self):
        """Verify that each call to create_guard returns a distinct instance."""
        from guardrails_ai_example import create_guard

        guard_a = create_guard()
        guard_b = create_guard()
        assert guard_a is not guard_b


class TestValidateResponse:
    """Tests for the validate_response function."""

    def test_validate_response_success_structure(self):
        """Verify that a successful validation returns the expected dict shape."""
        from guardrails_ai_example import validate_response

        mock_guard = MagicMock()
        mock_validated = MagicMock()
        mock_validated.validated_output = "Hello! How can I help you today?"
        mock_guard.return_value = mock_validated

        with patch("guardrails_ai_example.litellm"):
            result = validate_response(mock_guard, "Say hello politely.")

        assert result["success"] is True
        assert result["output"] == "Hello! How can I help you today?"
        assert result["error"] is None

    def test_validate_response_calls_guard_once(self):
        """Verify that the guard is called exactly once per invocation."""
        from guardrails_ai_example import validate_response

        mock_guard = MagicMock()
        mock_validated = MagicMock()
        mock_validated.validated_output = "A safe response."
        mock_guard.return_value = mock_validated

        with patch("guardrails_ai_example.litellm"):
            validate_response(mock_guard, "Hello")

        mock_guard.assert_called_once()

    def test_validate_response_on_exception_returns_failure(self):
        """Verify that exceptions are caught and returned as a failure result."""
        from guardrails_ai_example import validate_response

        mock_guard = MagicMock()
        mock_guard.side_effect = Exception("Guardrail validation failed")

        with patch("guardrails_ai_example.litellm"):
            result = validate_response(mock_guard, "Test prompt")

        assert result["success"] is False
        assert result["output"] is None
        assert "Guardrail validation failed" in result["error"]

    def test_validate_response_always_has_required_keys(self):
        """Verify that all required keys are present regardless of outcome."""
        from guardrails_ai_example import validate_response

        mock_guard = MagicMock()
        mock_guard.side_effect = RuntimeError("Unexpected error")

        with patch("guardrails_ai_example.litellm"):
            result = validate_response(mock_guard, "Any prompt")

        assert "success" in result
        assert "output" in result
        assert "error" in result

    def test_validate_response_error_contains_message(self):
        """Verify that the error field captures the original exception message."""
        from guardrails_ai_example import validate_response

        mock_guard = MagicMock()
        mock_guard.side_effect = ValueError("ProfanityFree check failed")

        with patch("guardrails_ai_example.litellm"):
            result = validate_response(mock_guard, "Some input")

        assert "ProfanityFree check failed" in result["error"]
