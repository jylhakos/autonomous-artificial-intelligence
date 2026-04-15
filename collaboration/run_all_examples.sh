#!/bin/bash
# Run All Collaboration Examples
# ==============================
# This script runs all AI agent collaboration examples sequentially.

set -e  # Exit on error

echo "======================================================================"
echo "COLLABORATIVE AI AGENTS - RUNNING ALL EXAMPLES"
echo "======================================================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check API key
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set"
    echo ""
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    exit 1
fi

echo " Virtual environment active"
echo " API key configured"
echo ""

# Run examples
examples=(
    "autogen_hello_agent.py:Basic Hello Agent"
    "autogen_multi_agent_collaboration.py:Multi-Agent Collaboration"
    "hybrid_agent_collaboration.py:Hybrid Agent Collaboration"
)

for example in "${examples[@]}"; do
    IFS=':' read -r file description <<< "$example"
    
    if [[ ! -f "$file" ]]; then
        echo " $file not found, skipping..."
        continue
    fi
    
    echo "======================================================================"
    echo "Running: $description"
    echo "File: $file"
    echo "======================================================================"
    echo ""
    
    python "$file"
    
    echo ""
    echo "----------------------------------------------------------------------"
    echo " $description completed"
    echo "----------------------------------------------------------------------"
    echo ""
    echo "Press Enter to continue to next example (or Ctrl+C to stop)..."
    read -r
    echo ""
done

echo "======================================================================"
echo "ALL EXAMPLES COMPLETED SUCCESSFULLY"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Review the output above to see how agents collaborated"
echo "2. Run test suite: python test_collaboration_suite.py"
echo "3. Modify examples to experiment with different configurations"
echo "4. See README.md for testing collaboration methods"
