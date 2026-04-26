# conftest.py - shared pytest configuration and fixtures

# Add the project root to the Python path so that test files can import
# the example scripts (guardrails_ai_example, pydantic_ai_example,
# llm_guard_example) without installing them as packages.

import sys
from pathlib import Path

# Insert the parent directory (security/) into sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
