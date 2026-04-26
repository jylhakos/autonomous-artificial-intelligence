"""
crewai_guardrails.py - CrewAI Multi-Agent Guardrails Demo

Demonstrates how to implement guardrails in a CrewAI multi-agent system:
  - Function-based guardrails: custom Python validation logic (PII, length, topics)
  - String-based guardrails: natural language compliance constraints
  - Combined validation pipeline: multiple checks chained together
  - Multi-agent crew: primary agent + compliance validator agent

Requirements:
  pip install crewai>=0.80.0

LLM Backend:
  Ollama (local LLM server) using the OpenAI-compatible /v1 endpoint.
  Ensure Ollama is running before executing the full crew:
    ollama serve
    ollama pull llama3

Run modes:
  # Standalone guardrail demo (no LLM required):
  python crewai_guardrails.py --demo

  # Full crew execution (Ollama required):
  python crewai_guardrails.py

References:
  - CrewAI GitHub: https://github.com/crewaiinc/crewai
  - CrewAI Hallucination Guardrail: https://docs.crewai.com/en/enterprise/features/hallucination-guardrail
  - AWS: Build safe and responsible generative AI applications with guardrails:
    https://aws.amazon.com/blogs/machine-learning/build-safe-and-responsible-generative-ai-applications-with-guardrails/
  - AWS: Build agentic AI solutions with DeepSeek-R1, CrewAI, and Amazon SageMaker AI:
    https://aws.amazon.com/blogs/machine-learning/build-agentic-ai-solutions-with-deepseek-r1-crewai-and-amazon-sagemaker-ai/
"""

import re
import sys

# ─── Ollama LLM Configuration ─────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "ollama/llama3"


# ─── Function-Based Guardrails ────────────────────────────────────────────────
#
# Function-based guardrails are plain Python callables.
# They receive the TaskOutput object and return a (bool, str) tuple:
#   (True,  "Reason passed")   → output is accepted
#   (False, "Reason failed")   → output is rejected; CrewAI re-prompts the agent
#
# This gives full control over validation logic — no LLM call needed.


def pii_guardrail(output) -> tuple:
    """
    Detect and block Personally Identifiable Information (PII) in agent output.

    Scans for email addresses, phone numbers, Social Security Numbers, and
    credit card numbers using regular expressions.

    Args:
        output: CrewAI TaskOutput object (has a .raw attribute with the text).

    Returns:
        (bool, str): (is_valid, feedback) tuple.
    """
    text = output.raw if hasattr(output, "raw") else str(output)

    pii_patterns = {
        "email address": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "phone number": r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit card number": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    found = []
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, text):
            found.append(pii_type)

    if found:
        return (
            False,
            f"Output contains PII ({', '.join(found)}). "
            "Remove all sensitive personal information and regenerate.",
        )

    return True, "No PII detected. Output is safe."


def length_guardrail(output) -> tuple:
    """
    Validate that the output meets minimum and maximum word count requirements.

    Ensures responses are meaningful (at least 10 words) but concise (max 500 words).

    Args:
        output: CrewAI TaskOutput object.

    Returns:
        (bool, str): (is_valid, feedback) tuple.
    """
    text = output.raw if hasattr(output, "raw") else str(output)
    word_count = len(text.split())

    if word_count < 10:
        return (
            False,
            f"Output is too short ({word_count} words). "
            "Provide a more detailed and complete response.",
        )

    if word_count > 500:
        return (
            False,
            f"Output is too long ({word_count} words). "
            "Please be more concise — maximum 500 words.",
        )

    return True, f"Output length is acceptable ({word_count} words)."


def topic_guardrail(output) -> tuple:
    """
    Block output containing restricted or harmful topics.

    Scans the agent response for banned keywords associated with cyberattacks,
    exploitation, or malware. If detected, the task is rejected and the agent
    is forced to regenerate a compliant response.

    Args:
        output: CrewAI TaskOutput object.

    Returns:
        (bool, str): (is_valid, feedback) tuple.
    """
    text = output.raw.lower() if hasattr(output, "raw") else str(output).lower()

    banned_keywords = [
        "hack",
        "exploit",
        "malware",
        "ransomware",
        "phishing",
        "bypass security",
        "sql injection",
        "xss attack",
        "ddos",
        "zero-day",
        "rootkit",
    ]

    detected = [kw for kw in banned_keywords if kw in text]

    if detected:
        return (
            False,
            f"Output references restricted topics: {', '.join(detected)}. "
            "This content violates the safety policy and cannot be returned to the user.",
        )

    return True, "Content passes topic filter. No restricted topics detected."


def combined_guardrail(output) -> tuple:
    """
    Pipeline guardrail: runs PII, length, and topic checks sequentially.

    Implements a Defense-in-Depth validation strategy. Each guardrail is
    applied in order — the first failure short-circuits the pipeline and
    returns an actionable rejection message to the agent.

    This is the recommended pattern for production CrewAI deployments where
    multiple constraints must all be satisfied before a response is accepted.

    Args:
        output: CrewAI TaskOutput object.

    Returns:
        (bool, str): (is_valid, feedback) tuple.
    """
    for guardrail_fn in [pii_guardrail, length_guardrail, topic_guardrail]:
        is_valid, feedback = guardrail_fn(output)
        if not is_valid:
            return False, feedback

    return True, "Output passed all guardrail checks (PII, length, topic)."


# ─── CrewAI Crew Setup ────────────────────────────────────────────────────────


def create_ollama_llm():
    """
    Create an Ollama LLM instance using the OpenAI-compatible /v1 endpoint.

    CrewAI's LLM class accepts any OpenAI-compatible API, making it easy to
    use locally-hosted models via Ollama without an API key.

    Returns:
        crewai.LLM: Configured LLM instance pointing to local Ollama.
    """
    from crewai import LLM

    return LLM(
        model=OLLAMA_MODEL,
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
    )


def create_agents(llm):
    """
    Create the primary AI agent and a compliance validator agent.

    The two-agent architecture demonstrates a common guardrail pattern:
      - Primary agent: generates the content
      - Compliance agent: audits the content before it is finalized

    Args:
        llm: CrewAI LLM instance to assign to both agents.

    Returns:
        tuple[Agent, Agent]: (primary_agent, compliance_agent)
    """
    from crewai import Agent

    primary_agent = Agent(
        role="AI Security Assistant",
        goal=(
            "Provide accurate, safe, and helpful information about AI security topics. "
            "Never include personal data, harmful instructions, or off-topic content."
        ),
        backstory=(
            "You are a knowledgeable AI security expert. You explain concepts clearly "
            "and professionally without revealing sensitive or harmful information. "
            "You refuse to assist with any malicious or unethical activities."
        ),
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    compliance_agent = Agent(
        role="Compliance Validator",
        goal=(
            "Review AI-generated content and confirm it meets safety, privacy, "
            "and compliance standards before it reaches end-users."
        ),
        backstory=(
            "You are a strict compliance officer responsible for auditing AI output. "
            "You check that responses are free from PII, stay within approved topic "
            "boundaries, and meet quality standards. You provide clear feedback "
            "when content fails any compliance check."
        ),
        llm=llm,
        verbose=True,
        max_iter=2,
    )

    return primary_agent, compliance_agent


def create_tasks(primary_agent, compliance_agent) -> list:
    """
    Create CrewAI tasks with function-based and string-based guardrails.

    Task 1 (answer_task):
        Uses the combined_guardrail function — a Python-level validation
        pipeline (PII + length + topic). If the agent's output fails, CrewAI
        automatically re-prompts the agent with the failure feedback.

    Task 2 (compliance_task):
        Uses a string-based guardrail — a natural language constraint that
        CrewAI evaluates via an internal LLM call. This is useful when the
        validation rule is easier to describe in plain English than in code.

    Args:
        primary_agent: The AI Security Assistant agent.
        compliance_agent: The Compliance Validator agent.

    Returns:
        list[Task]: Ordered list of tasks for sequential crew execution.
    """
    from crewai import Task

    # Task 1: Generate content — protected by function-based combined guardrail
    answer_task = Task(
        description=(
            "Explain what 'Defense in Depth' means in the context of AI agent security. "
            "Describe at least three layers of protection and why each layer matters. "
            "Keep the explanation professional and suitable for a technical audience."
        ),
        expected_output=(
            "A clear, concise explanation of Defense in Depth for AI security "
            "(100 to 300 words), covering input-level, agent-level, and "
            "infrastructure-level protections."
        ),
        agent=primary_agent,
        guardrail=combined_guardrail,  # Function-based: PII + length + topic checks
    )

    # Task 2: Compliance review — protected by string-based (natural language) guardrail
    compliance_task = Task(
        description=(
            "Review the previous explanation about Defense in Depth and produce a "
            "compliance assessment. Confirm whether the content is safe, accurate, "
            "free from PII, and appropriate for a general technical audience."
        ),
        expected_output=(
            "A brief compliance summary (50 to 150 words) stating: "
            "(1) whether the content passed safety checks, "
            "(2) any issues found, and "
            "(3) a final pass/fail verdict."
        ),
        agent=compliance_agent,
        context=[answer_task],
        # String-based guardrail: natural language rule evaluated by an internal LLM call
        guardrail=(
            "Ensure the compliance review does not reveal any personal data, "
            "system vulnerabilities, internal credentials, or details that could "
            "assist a malicious actor."
        ),
    )

    return [answer_task, compliance_task]


# ─── Main Crew Execution ──────────────────────────────────────────────────────


def run_guardrail_demo() -> dict:
    """
    Run the full CrewAI multi-agent guardrails demonstration.

    Creates a sequential crew with:
      - A primary agent that generates an AI security explanation
      - A compliance agent that audits the response
      - Function-based and string-based guardrails on each task

    Returns:
        dict: {success: bool, task_outputs: list[str], final_output: str}
    """
    from crewai import Crew, Process

    try:
        llm = create_ollama_llm()
        primary_agent, compliance_agent = create_agents(llm)
        tasks = create_tasks(primary_agent, compliance_agent)

        crew = Crew(
            agents=[primary_agent, compliance_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        return {
            "success": True,
            "task_outputs": [
                task.output.raw if task.output else "" for task in tasks
            ],
            "final_output": str(result),
        }

    except Exception as exc:
        return {
            "success": False,
            "task_outputs": [],
            "final_output": f"Error: {exc}",
        }


# ─── Standalone Guardrail Demo (no LLM required) ─────────────────────────────


def demo_guardrails_standalone() -> None:
    """
    Demonstrate each guardrail function in isolation without running a crew.

    This mode is useful for quickly verifying guardrail logic or running tests
    in an environment where Ollama is not available.
    """

    class MockOutput:
        """Minimal stand-in for crewai.tasks.task_output.TaskOutput."""

        def __init__(self, text: str):
            self.raw = text

    print("\n=== Standalone Guardrail Function Demonstrations ===\n")

    cases = [
        (
            "PII guardrail — email detected",
            pii_guardrail,
            "Please contact john.doe@example.com for support.",
        ),
        (
            "PII guardrail — clean text",
            pii_guardrail,
            "Defense in Depth applies multiple independent security controls.",
        ),
        (
            "Length guardrail — too short",
            length_guardrail,
            "Yes.",
        ),
        (
            "Length guardrail — acceptable",
            length_guardrail,
            (
                "Defense in Depth is a layered security strategy where multiple "
                "independent controls are applied at different levels of a system. "
                "If one control fails, others continue to provide protection."
            ),
        ),
        (
            "Topic guardrail — restricted keyword",
            topic_guardrail,
            "Here is how to exploit a SQL injection vulnerability in a database.",
        ),
        (
            "Topic guardrail — safe content",
            topic_guardrail,
            "Input validation is essential to prevent malicious data from reaching the model.",
        ),
        (
            "Combined guardrail — PII in otherwise valid text",
            combined_guardrail,
            (
                "Call our security team at 555-867-5309 for immediate assistance "
                "with your AI deployment questions."
            ),
        ),
        (
            "Combined guardrail — fully compliant text",
            combined_guardrail,
            (
                "Defense in Depth for AI agents involves applying security controls "
                "at the input layer to detect prompt injections, at the agent layer "
                "to enforce role-based access and tool restrictions, and at the "
                "infrastructure layer to isolate compute and restrict network egress. "
                "Each layer independently reduces risk so that a breach of one layer "
                "does not compromise the entire system."
            ),
        ),
    ]

    for label, fn, text in cases:
        is_valid, feedback = fn(MockOutput(text))
        status = "PASS" if is_valid else "FAIL"
        print(f"[{status}] {label}")
        print(f"       Feedback: {feedback}\n")

    print("=== Standalone Demo Complete ===\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--demo" in sys.argv:
        # Run standalone guardrail demonstrations — no Ollama required
        demo_guardrails_standalone()
    else:
        print(
            "\nStarting CrewAI Guardrails Demo with Ollama...\n"
            "  Make sure Ollama is running:    ollama serve\n"
            "  Make sure the model is pulled:  ollama pull llama3\n"
        )

        result = run_guardrail_demo()

        if result["success"]:
            print("\n=== Crew Completed Successfully ===")
            print("\nFinal Output:")
            print(result["final_output"])
        else:
            print("\n=== Crew Execution Failed ===")
            print(result["final_output"])
            sys.exit(1)
