#!/usr/bin/env python3
"""
Simple test runner for blog_agents.py
Tests basic functionality without requiring full LLM execution.
"""

import sys
import os


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from blog_agents import BlogAgentOrchestrator
        from crewai import Agent, Task, Crew, Process
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_class_instantiation():
    """Test that the orchestrator class can be instantiated."""
    print("\nTesting class instantiation...")
    try:
        from blog_agents import BlogAgentOrchestrator
        orchestrator = BlogAgentOrchestrator("Test Topic", "openai")
        assert orchestrator.topic == "Test Topic"
        assert orchestrator.model_provider == "openai"
        print("  ✓ Class instantiation successful")
        return True
    except Exception as e:
        print(f"  ✗ Instantiation failed: {e}")
        return False


def test_agent_creation():
    """Test that agents can be created."""
    print("\nTesting agent creation...")
    try:
        from blog_agents import BlogAgentOrchestrator
        orchestrator = BlogAgentOrchestrator("Test Topic", "openai")
        
        planner = orchestrator.create_planner_agent()
        writer = orchestrator.create_writer_agent()
        editor = orchestrator.create_editor_agent()
        
        assert planner is not None
        assert writer is not None
        assert editor is not None
        
        print("  ✓ All agents created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False


def test_task_creation():
    """Test that tasks can be created."""
    print("\nTesting task creation...")
    try:
        from blog_agents import BlogAgentOrchestrator
        orchestrator = BlogAgentOrchestrator("Test Topic", "openai")
        
        planner = orchestrator.create_planner_agent()
        planning_task = orchestrator.create_planning_task(planner)
        
        assert planning_task is not None
        assert hasattr(planning_task, 'description')
        assert hasattr(planning_task, 'expected_output')
        
        print("  ✓ Tasks created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Task creation failed: {e}")
        return False


def test_llm_config():
    """Test LLM configuration for different providers."""
    print("\nTesting LLM configuration...")
    try:
        from blog_agents import BlogAgentOrchestrator
        
        # Test OpenAI config
        orchestrator_openai = BlogAgentOrchestrator("Test", "openai")
        config_openai = orchestrator_openai.get_llm_config()
        
        # Test Ollama config
        orchestrator_ollama = BlogAgentOrchestrator("Test", "ollama")
        config_ollama = orchestrator_ollama.get_llm_config()
        assert config_ollama is not None
        assert "model" in config_ollama
        
        # Test Bedrock config
        orchestrator_bedrock = BlogAgentOrchestrator("Test", "bedrock")
        config_bedrock = orchestrator_bedrock.get_llm_config()
        assert config_bedrock is not None
        assert "model" in config_bedrock
        
        print("  ✓ LLM configurations valid")
        return True
    except Exception as e:
        print(f"  ✗ LLM config test failed: {e}")
        return False


def test_environment():
    """Test environment variables and setup."""
    print("\nTesting environment...")
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 9):
        issues.append("Python 3.9+ required")
    
    # Check virtual environment
    if not hasattr(sys, 'real_prefix') and not (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        issues.append("Virtual environment not active")
    
    # Check for LLM provider
    has_provider = False
    if os.getenv('OPENAI_API_KEY'):
        has_provider = True
        print("  ✓ OpenAI API key found")
    if os.getenv('AWS_REGION'):
        has_provider = True
        print("  ✓ AWS region configured")
    if os.path.exists('/usr/local/bin/ollama'):
        has_provider = True
        print("  ✓ Ollama detected")
    
    if not has_provider:
        issues.append("No LLM provider configured (set OPENAI_API_KEY or install Ollama)")
    
    if issues:
        print(f"  ⚠ Environment issues: {', '.join(issues)}")
        return False
    else:
        print("  ✓ Environment configured correctly")
        return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Blog Agent Orchestrator - Unit Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_class_instantiation,
        test_agent_creation,
        test_task_creation,
        test_llm_config,
        test_environment,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
