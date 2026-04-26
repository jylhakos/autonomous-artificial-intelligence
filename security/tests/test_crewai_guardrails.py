"""
tests/test_crewai_guardrails.py

Unit tests for crewai_guardrails.py.

All tests use mock objects so that neither Ollama nor the crewai package
is required to run the test suite. Import of crewai is deferred inside
the functions under test, so we mock only what is actually imported.

Run:
    pytest tests/test_crewai_guardrails.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockOutput:
    """Minimal stand-in for crewai.tasks.task_output.TaskOutput."""

    def __init__(self, text: str):
        self.raw = text


# ---------------------------------------------------------------------------
# pii_guardrail
# ---------------------------------------------------------------------------

class TestPiiGuardrail:
    """Tests for the pii_guardrail function."""

    def test_detects_email(self):
        from crewai_guardrails import pii_guardrail

        result, feedback = pii_guardrail(MockOutput("Send to admin@example.com"))
        assert result is False
        assert "email address" in feedback

    def test_detects_phone_number(self):
        from crewai_guardrails import pii_guardrail

        result, feedback = pii_guardrail(MockOutput("Call 555-867-5309 now."))
        assert result is False
        assert "phone number" in feedback

    def test_detects_ssn(self):
        from crewai_guardrails import pii_guardrail

        result, feedback = pii_guardrail(MockOutput("SSN: 123-45-6789"))
        assert result is False
        assert "SSN" in feedback

    def test_detects_credit_card(self):
        from crewai_guardrails import pii_guardrail

        result, feedback = pii_guardrail(MockOutput("Card: 4111 1111 1111 1111"))
        assert result is False
        assert "credit card" in feedback

    def test_passes_clean_text(self):
        from crewai_guardrails import pii_guardrail

        result, feedback = pii_guardrail(
            MockOutput("Defense in Depth applies multiple independent controls.")
        )
        assert result is True
        assert "No PII detected" in feedback

    def test_uses_raw_attribute(self):
        """Guardrail reads from output.raw, not str(output)."""
        from crewai_guardrails import pii_guardrail

        output = MockOutput("safe text without any personal information here")
        result, _ = pii_guardrail(output)
        assert result is True

    def test_fallback_to_str_when_no_raw(self):
        """If output has no .raw, str(output) is used."""
        from crewai_guardrails import pii_guardrail

        output = MagicMock(spec=[])  # no 'raw' attribute
        output.__str__ = lambda self: "contact us at test@mail.org"
        result, feedback = pii_guardrail(output)
        assert result is False


# ---------------------------------------------------------------------------
# length_guardrail
# ---------------------------------------------------------------------------

class TestLengthGuardrail:
    """Tests for the length_guardrail function."""

    def test_rejects_too_short(self):
        from crewai_guardrails import length_guardrail

        result, feedback = length_guardrail(MockOutput("Yes."))
        assert result is False
        assert "too short" in feedback

    def test_rejects_too_long(self):
        from crewai_guardrails import length_guardrail

        long_text = " ".join(["word"] * 501)
        result, feedback = length_guardrail(MockOutput(long_text))
        assert result is False
        assert "too long" in feedback

    def test_accepts_acceptable_length(self):
        from crewai_guardrails import length_guardrail

        text = " ".join(["word"] * 50)
        result, feedback = length_guardrail(MockOutput(text))
        assert result is True
        assert "acceptable" in feedback

    def test_boundary_exactly_ten_words(self):
        from crewai_guardrails import length_guardrail

        text = " ".join(["word"] * 10)
        result, _ = length_guardrail(MockOutput(text))
        assert result is True

    def test_boundary_exactly_500_words(self):
        from crewai_guardrails import length_guardrail

        text = " ".join(["word"] * 500)
        result, _ = length_guardrail(MockOutput(text))
        assert result is True

    def test_boundary_nine_words_fails(self):
        from crewai_guardrails import length_guardrail

        text = " ".join(["word"] * 9)
        result, _ = length_guardrail(MockOutput(text))
        assert result is False


# ---------------------------------------------------------------------------
# topic_guardrail
# ---------------------------------------------------------------------------

class TestTopicGuardrail:
    """Tests for the topic_guardrail function."""

    def test_blocks_exploit_keyword(self):
        from crewai_guardrails import topic_guardrail

        result, feedback = topic_guardrail(
            MockOutput("Here is how to exploit this vulnerability.")
        )
        assert result is False
        assert "exploit" in feedback

    def test_blocks_malware_keyword(self):
        from crewai_guardrails import topic_guardrail

        result, feedback = topic_guardrail(MockOutput("Install this malware payload."))
        assert result is False
        assert "malware" in feedback

    def test_blocks_sql_injection(self):
        from crewai_guardrails import topic_guardrail

        result, feedback = topic_guardrail(
            MockOutput("Demonstrate sql injection on the login form.")
        )
        assert result is False

    def test_passes_safe_security_content(self):
        from crewai_guardrails import topic_guardrail

        result, feedback = topic_guardrail(
            MockOutput(
                "Input validation is essential to prevent malicious data from "
                "reaching the model layer."
            )
        )
        assert result is True
        assert "No restricted topics" in feedback

    def test_case_insensitive(self):
        from crewai_guardrails import topic_guardrail

        result, _ = topic_guardrail(MockOutput("Deploy RANSOMWARE to all targets."))
        assert result is False


# ---------------------------------------------------------------------------
# combined_guardrail
# ---------------------------------------------------------------------------

class TestCombinedGuardrail:
    """Tests for the combined_guardrail pipeline function."""

    def test_fails_on_pii_first(self):
        from crewai_guardrails import combined_guardrail

        result, feedback = combined_guardrail(
            MockOutput("Contact us at user@example.com for help.")
        )
        assert result is False
        assert "PII" in feedback

    def test_fails_on_short_output(self):
        from crewai_guardrails import combined_guardrail

        result, feedback = combined_guardrail(MockOutput("OK."))
        # Short text fails length check; no PII so PII passes first
        assert result is False
        assert "short" in feedback

    def test_fails_on_banned_topic(self):
        from crewai_guardrails import combined_guardrail

        # Long enough, no PII, but contains banned keyword
        text = " ".join(["safe"] * 20) + " exploit this vulnerability"
        result, feedback = combined_guardrail(MockOutput(text))
        assert result is False
        assert "exploit" in feedback

    def test_passes_fully_compliant_text(self):
        from crewai_guardrails import combined_guardrail

        text = (
            "Defense in Depth for AI agents involves applying security controls "
            "at the input layer to detect prompt injections, at the agent layer "
            "to enforce role-based access and tool restrictions, and at the "
            "infrastructure layer to isolate compute and restrict network egress. "
            "Each layer independently reduces risk."
        )
        result, feedback = combined_guardrail(MockOutput(text))
        assert result is True
        assert "passed all guardrail checks" in feedback

    def test_returns_tuple(self):
        from crewai_guardrails import combined_guardrail

        output = combined_guardrail(MockOutput("Hello " * 15))
        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[0], bool)
        assert isinstance(output[1], str)


# ---------------------------------------------------------------------------
# demo_guardrails_standalone
# ---------------------------------------------------------------------------

class TestDemoGuardrailsStandalone:
    """Tests for the standalone demo function (output and no crashes)."""

    def test_runs_without_error(self, capsys):
        from crewai_guardrails import demo_guardrails_standalone

        demo_guardrails_standalone()
        captured = capsys.readouterr()
        assert "Standalone Guardrail Function Demonstrations" in captured.out
        assert "Standalone Demo Complete" in captured.out

    def test_shows_pass_and_fail(self, capsys):
        from crewai_guardrails import demo_guardrails_standalone

        demo_guardrails_standalone()
        captured = capsys.readouterr()
        assert "[PASS]" in captured.out
        assert "[FAIL]" in captured.out


# ---------------------------------------------------------------------------
# create_ollama_llm (import guard — does not call Ollama)
# ---------------------------------------------------------------------------

class TestCreateOllamaLlm:
    """Tests that create_ollama_llm builds the LLM with correct parameters."""

    def test_llm_model_and_base_url(self):
        mock_llm_cls = MagicMock()

        with patch.dict("sys.modules", {"crewai": MagicMock(LLM=mock_llm_cls)}):
            from importlib import reload
            import crewai_guardrails as cg
            reload(cg)
            cg.create_ollama_llm()

        mock_llm_cls.assert_called_once()
        call_kwargs = mock_llm_cls.call_args.kwargs
        assert call_kwargs["model"] == "ollama/llama3"
        assert "11434" in call_kwargs["base_url"]
        assert call_kwargs["api_key"] == "ollama"


# ---------------------------------------------------------------------------
# run_guardrail_demo (mocked crew)
# ---------------------------------------------------------------------------

class TestRunGuardrailDemo:
    """Tests for run_guardrail_demo with a fully mocked CrewAI stack."""

    def _make_crewai_mock(self):
        """Build a minimal mock of the crewai module."""
        mock_output = MagicMock()
        mock_output.raw = "Mocked task output"

        mock_task = MagicMock()
        mock_task.output = mock_output

        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.return_value = "Final mocked result"

        mock_crewai = MagicMock()
        mock_crewai.LLM = MagicMock(return_value=MagicMock())
        mock_crewai.Agent = MagicMock(return_value=MagicMock())
        mock_crewai.Task = MagicMock(return_value=mock_task)
        mock_crewai.Crew = MagicMock(return_value=mock_crew_instance)
        mock_crewai.Process = MagicMock()
        mock_crewai.Process.sequential = "sequential"

        return mock_crewai

    def test_returns_success_dict(self):
        mock_crewai = self._make_crewai_mock()

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
            from importlib import reload
            import crewai_guardrails as cg
            reload(cg)
            result = cg.run_guardrail_demo()

        assert result["success"] is True
        assert "final_output" in result
        assert "task_outputs" in result

    def test_handles_exception_gracefully(self):
        mock_crewai = MagicMock()
        mock_crewai.LLM.side_effect = RuntimeError("Ollama not running")

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
            from importlib import reload
            import crewai_guardrails as cg
            reload(cg)
            result = cg.run_guardrail_demo()

        assert result["success"] is False
        assert "Error" in result["final_output"]

    def test_result_always_has_required_keys(self):
        mock_crewai = self._make_crewai_mock()

        with patch.dict("sys.modules", {"crewai": mock_crewai}):
            from importlib import reload
            import crewai_guardrails as cg
            reload(cg)
            result = cg.run_guardrail_demo()

        for key in ("success", "task_outputs", "final_output"):
            assert key in result
