#!/usr/bin/env python3
"""
Environment Setup Checker
=========================
This script verifies that your environment is correctly configured
for running the collaborative AI agent examples.

It checks:
1. Python version
2. Virtual environment
3. Required packages
4. API keys
5. Network connectivity (optional)
"""

import sys
import os
import importlib.util


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False


def check_virtual_env():
    """Check if running in a virtual environment."""
    print("\nChecking virtual environment...")
    in_venv = hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix
    if in_venv:
        print(f"  ✓ Virtual environment active: {sys.prefix}")
        return True
    else:
        print("  ⚠ Not in a virtual environment (recommended but not required)")
        return True


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        print(f"  ✓ {package_name} is installed")
        return True
    else:
        print(f"  ✗ {package_name} is NOT installed")
        return False


def check_packages():
    """Check if required packages are installed."""
    print("\nChecking required packages...")
    
    packages = [
        ("autogen-agentchat", "autogen_agentchat"),
        ("autogen-ext", "autogen_ext"),
        ("autogen-core", "autogen_core"),
    ]
    
    all_installed = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_installed = False
    
    return all_installed


def check_optional_packages():
    """Check optional packages."""
    print("\nChecking optional packages...")
    
    packages = [
        ("agent-framework-claude", "agent_framework_claude"),
    ]
    
    for pkg_name, import_name in packages:
        check_package(pkg_name, import_name)


def check_api_keys():
    """Check if API keys are set."""
    print("\nChecking API keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        masked_key = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
        print(f"  ✓ OPENAI_API_KEY is set ({masked_key})")
        openai_ok = True
    else:
        print("  ✗ OPENAI_API_KEY is NOT set (required for AutoGen examples)")
        print("    Set it with: export OPENAI_API_KEY='your-key-here'")
        openai_ok = False
    
    if anthropic_key:
        masked_key = anthropic_key[:8] + "..." + anthropic_key[-4:] if len(anthropic_key) > 12 else "***"
        print(f"  ✓ ANTHROPIC_API_KEY is set ({masked_key})")
    else:
        print("  ⚠ ANTHROPIC_API_KEY is NOT set (optional, for Claude examples)")
        print("    Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    
    return openai_ok


def check_examples_exist():
    """Check if example files exist."""
    print("\nChecking example files...")
    
    examples = [
        "autogen_hello_agent.py",
        "autogen_multi_agent_collaboration.py",
        "claude_agent_basic.py",
        "hybrid_agent_collaboration.py",
    ]
    
    all_exist = True
    for example in examples:
        path = os.path.join("examples", example)
        if os.path.exists(path):
            print(f"  ✓ {example}")
        else:
            print(f"  ✗ {example} NOT FOUND")
            all_exist = False
    
    return all_exist


def print_summary(checks):
    """Print summary of all checks."""
    print("\n" + "=" * 70)
    print("SETUP CHECK SUMMARY")
    print("=" * 70)
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n✓ All critical checks passed!")
        print("\nYou're ready to run the examples:")
        print("  python examples/autogen_hello_agent.py")
        print("  python examples/autogen_multi_agent_collaboration.py")
        print("  python examples/hybrid_agent_collaboration.py")
    else:
        print("\n⚠ Some checks failed. Please address the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -U 'autogen-agentchat' 'autogen-ext[openai]'")
        print("  pip install agent-framework-claude --pre")
        print("\nTo set API keys:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")


def main():
    """Run all checks."""
    print("=" * 70)
    print("COLLABORATIVE AI AGENTS - ENVIRONMENT SETUP CHECKER")
    print("=" * 70)
    
    checks = {
        "python": check_python_version(),
        "venv": check_virtual_env(),
        "packages": check_packages(),
        "api_keys": check_api_keys(),
        "examples": check_examples_exist(),
    }
    
    check_optional_packages()
    print_summary(checks)
    
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
