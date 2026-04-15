"""
Main Entry Point for Agent Evaluation Examples

This script provides a command-line interface to run different
agent evaluation examples and demonstrations.
"""

import os
import sys
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    Autonomous AI Agent Evaluation with LangChain & LangSmith    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check if required environment variables are set."""
    required_vars = {
        "LANGCHAIN_API_KEY": "LangSmith API key (for tracing and evaluation)",
        "ANTHROPIC_API_KEY": "Anthropic API key (for Claude models)",
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  - {var}: {description}")
    
    if missing_vars:
        print("⚠️  Missing required environment variables:\n")
        print("\n".join(missing_vars))
        print("\nPlease set these variables in your .env file")
        print("See .env.example for template\n")
        return False
    
    return True


def show_menu():
    """Display the main menu."""
    menu = """
Available Examples:
───────────────────────────────────────────────────────────────────

1. Basic Agent
   Create and run a simple AI agent with tool-calling capabilities
   
2. Agent with Tracing
   Run an agent with LangSmith observability and tracing enabled
   
3. Agent Evaluation (Offline)
   Evaluate agent performance using golden datasets and custom evaluators
   
4. Online Evaluation & Monitoring
   Simulate production monitoring and real-time evaluation
   
5. Run All Examples
   Execute all examples in sequence
   
0. Exit

───────────────────────────────────────────────────────────────────
    """
    print(menu)


def run_basic_agent():
    """Run the basic agent example."""
    print("\n" + "=" * 70)
    print("Running: Basic Agent Example")
    print("=" * 70 + "\n")
    
    try:
        from src.basic_agent import main
        main()
    except Exception as e:
        print(f"Error: {str(e)}")


def run_traced_agent():
    """Run the agent with tracing example."""
    print("\n" + "=" * 70)
    print("Running: Agent with LangSmith Tracing")
    print("=" * 70 + "\n")
    
    try:
        from src.agent_with_tracing import main
        main()
    except Exception as e:
        print(f"Error: {str(e)}")


def run_evaluation():
    """Run the agent evaluation example."""
    print("\n" + "=" * 70)
    print("Running: Agent Evaluation (Offline)")
    print("=" * 70 + "\n")
    
    try:
        from src.agent_evaluation import main
        main()
    except Exception as e:
        print(f"Error: {str(e)}")


def run_online_evaluation():
    """Run the online evaluation example."""
    print("\n" + "=" * 70)
    print("Running: Online Evaluation & Monitoring")
    print("=" * 70 + "\n")
    
    try:
        from src.online_evaluation import main
        main()
    except Exception as e:
        print(f"Error: {str(e)}")


def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        ("Basic Agent", run_basic_agent),
        ("Agent with Tracing", run_traced_agent),
        ("Agent Evaluation", run_evaluation),
        ("Online Evaluation", run_online_evaluation),
    ]
    
    for name, func in examples:
        print("\n" + "=" * 70)
        print(f"Starting: {name}")
        print("=" * 70)
        
        try:
            func()
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
        
        input("\nPress Enter to continue to next example...")


def main():
    """Main application entry point."""
    print_banner()
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    print("✓ Environment variables configured\n")
    
    # Main application loop
    while True:
        show_menu()
        
        try:
            choice = input("Select an option (0-5): ").strip()
            
            if choice == "0":
                print("\nThank you for using the Agent Evaluation Framework!")
                print("View your traces at: https://smith.langchain.com/\n")
                break
            
            elif choice == "1":
                run_basic_agent()
            
            elif choice == "2":
                run_traced_agent()
            
            elif choice == "3":
                run_evaluation()
            
            elif choice == "4":
                run_online_evaluation()
            
            elif choice == "5":
                run_all_examples()
            
            else:
                print("\n⚠️  Invalid option. Please select 0-5.\n")
                continue
            
            input("\nPress Enter to return to menu...")
            print("\n" * 2)
            
        except KeyboardInterrupt:
            print("\n\nExiting...\n")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
