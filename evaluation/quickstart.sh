#!/bin/bash

# Quick Start Script for Agent Evaluation Framework
# This script helps you set up the environment and run examples

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   Autonomous AI Agent Evaluation - Quick Start                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and add your API keys:"
    echo "   - LANGCHAIN_API_KEY (get from https://smith.langchain.com/)"
    echo "   - ANTHROPIC_API_KEY (get from https://console.anthropic.com/)"
    echo ""
    echo "After adding your API keys, run this script again or execute:"
    echo "   python main.py"
    echo ""
    exit 0
fi

# Check if API keys are set
source .env

if [ -z "$LANGCHAIN_API_KEY" ] || [ "$LANGCHAIN_API_KEY" = "your_langsmith_api_key_here" ]; then
    echo "⚠️  LANGCHAIN_API_KEY not configured in .env file"
    echo "   Get your key from: https://smith.langchain.com/"
    echo ""
fi

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not configured in .env file"
    echo "   Get your key from: https://console.anthropic.com/"
    echo ""
fi

# Run main application
echo "🚀 Launching application..."
echo ""
python main.py

# Deactivate virtual environment on exit
deactivate
