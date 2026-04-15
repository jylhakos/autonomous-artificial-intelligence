"""
Comprehensive Collaboration Test Suite
=======================================
Run all collaboration tests in sequence to validate agent collaboration.

This suite runs:
1. Code Collaboration Test - Multi-agent code generation and review
2. Human-in-Loop Test - Interactive feedback and adaptation
3. Basic Collaboration Test - Simple multi-agent interaction

Usage:
    python test_collaboration_suite.py
"""

import asyncio
import os
import sys


async def run_all_tests():
    """Run complete collaboration test suite."""
    
    # Check API key first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("="*70)
        print("ERROR: OPENAI_API_KEY NOT SET")
        print("="*70)
        print("\nPlease set your API key before running tests:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nGet your key from: https://platform.openai.com/api-keys")
        return
    
    print("\n" + "="*70)
    print("AUTONOMOUS AI COLLABORATION TEST SUITE")
    print("="*70)
    print("\nThis suite will test various collaboration patterns:")
    print("  1. Multi-agent code generation and review")
    print("  2. Human-in-the-loop feedback integration")
    print("  3. Agent turn-taking and communication")
    print("\n" + "-"*70)
    
    # Import test modules
    try:
        from test_code_collaboration import test_code_collaboration
        from test_human_feedback import test_human_in_loop
    except ImportError as e:
        print(f"ERROR: Could not import test modules: {e}")
        print("Make sure test_code_collaboration.py and test_human_feedback.py exist")
        return
    
    tests = [
        ("Code Collaboration Test", test_code_collaboration),
        ("Human-in-Loop Test", test_human_in_loop),
    ]
    
    results = []
    
    for i, (test_name, test_func) in enumerate(tests, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}/{len(tests)}: {test_name}")
        print("="*70)
        
        try:
            await test_func()
            results.append((test_name, "PASSED"))
            print(f"\n {test_name} COMPLETED")
        except Exception as e:
            print(f"\n {test_name} FAILED")
            print(f"Error: {e}")
            results.append((test_name, "FAILED"))
        
        if i < len(tests):
            print("\n" + "-"*70)
            print("Waiting 2 seconds before next test...")
            print("-"*70)
            await asyncio.sleep(2)
    
    # Final Summary
    print("\n\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    for name, status in results:
        symbol = " " if status == "PASSED" else " "
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n All collaboration tests passed successfully!")
        print(" Agent collaboration is working as expected.")
    else:
        print(f"\n {total - passed} test(s) failed. Please review the errors above.")
    
    print("\nFor more details on collaboration testing, see:")
    print("  - README.md (Testing Agent Collaboration section)")
    print("  - test_code_collaboration.py (source code)")
    print("  - test_human_feedback.py (source code)")


def main():
    """Entry point for the test suite."""
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
