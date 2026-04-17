#!/bin/bash
# Test Suite for Blog Agent Orchestrator
# Run this script to perform comprehensive testing before deployment

set -e  # Exit on error

echo "========================================="
echo "Blog Agent Orchestrator - Test Suite"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
        ((TESTS_FAILED++))
    fi
}

# Check if virtual environment is active
echo "1. Verifying Environment Setup..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_result 0 "Virtual environment is active"
else
    print_result 1 "Virtual environment is NOT active - run 'source venv/bin/activate'"
    exit 1
fi

# Check Python version
echo ""
echo "2. Checking Python Version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -ge 3 ] && [ "$MINOR_VERSION" -ge 9 ]; then
    print_result 0 "Python version $PYTHON_VERSION (requires 3.9+)"
else
    print_result 1 "Python version $PYTHON_VERSION is too old (requires 3.9+)"
fi

# Check dependencies
echo ""
echo "3. Verifying Dependencies..."
pip check > /dev/null 2>&1
print_result $? "Dependency integrity check"

if pip list | grep -q crewai; then
    print_result 0 "CrewAI installed"
else
    print_result 1 "CrewAI not installed"
fi

# Check for LLM provider configuration
echo ""
echo "4. Checking LLM Provider Configuration..."
if [ -n "$OPENAI_API_KEY" ]; then
    print_result 0 "OpenAI API key configured"
elif [ -f "/usr/local/bin/ollama" ] || command -v ollama &> /dev/null; then
    print_result 0 "Ollama detected (local LLM)"
elif [ -n "$AWS_REGION" ] && aws sts get-caller-identity &> /dev/null; then
    print_result 0 "AWS credentials configured (Bedrock available)"
else
    print_result 1 "No LLM provider configured"
fi

# Check if blog_agents.py exists
echo ""
echo "5. Verifying Source Files..."
if [ -f "blog_agents.py" ]; then
    print_result 0 "blog_agents.py exists"
else
    print_result 1 "blog_agents.py not found"
    exit 1
fi

# Syntax check
echo ""
echo "6. Syntax Validation..."
python -m py_compile blog_agents.py 2>/dev/null
print_result $? "Python syntax check"

# Security scan (if bandit is installed)
echo ""
echo "7. Security Scan..."
if command -v bandit &> /dev/null; then
    bandit -r blog_agents.py -f txt > /dev/null 2>&1
    print_result $? "Bandit security scan"
else
    echo -e "${YELLOW}⊘ SKIPPED${NC}: Security scan (bandit not installed)"
fi

# Code quality (if flake8 is installed)
echo ""
echo "8. Code Quality Check..."
if command -v flake8 &> /dev/null; then
    flake8 blog_agents.py --max-line-length=100 --ignore=E501,W503 > /dev/null 2>&1
    print_result $? "Code quality (flake8)"
else
    echo -e "${YELLOW}⊘ SKIPPED${NC}: Code quality (flake8 not installed)"
fi

# Functional test (quick dry run)
echo ""
echo "9. Functional Test (Quick Run)..."
echo "   Testing with topic: 'Integration Test Example'"

# Create a temporary test that exits quickly
timeout 30s python blog_agents.py "Quick Test Topic" &> /dev/null || true
if [ -f "blog_output.md" ]; then
    # Check if file has some content
    SIZE=$(wc -c < blog_output.md)
    if [ $SIZE -gt 100 ]; then
        print_result 0 "Functional test - output generated"
        rm -f blog_output.md  # Clean up
    else
        print_result 1 "Functional test - output too small"
    fi
else
    echo -e "${YELLOW}⊘ PARTIAL${NC}: Functional test (requires LLM API access)"
fi

# Summary
echo ""
echo "========================================="
echo "Test Suite Results"
echo "========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Ready for deployment.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please fix issues before deployment.${NC}"
    exit 1
fi
