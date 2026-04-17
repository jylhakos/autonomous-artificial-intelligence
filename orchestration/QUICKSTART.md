# Quick Start

## Setup Steps

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure LLM provider (choose one):**

   **OpenAI:**
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

   **Ollama (local, free):**
   ```bash
   # Install: https://ollama.ai
   ollama pull mistral
   ```

   **AWS Bedrock:**
   ```bash
   aws configure
   pip install boto3 langchain-aws
   ```

4. **Run tests before deployment:**
   ```bash
   # Quick unit tests
   python run_tests.py
   
   # Comprehensive test suite
   ./test_suite.sh
   ```

5. **Run the multi-agent pipeline:**
   ```bash
   python blog_agents.py
   ```

   Or with custom topic:
   ```bash
   python blog_agents.py "Your Custom Topic Here"
   ```

## Output

The pipeline generates `blog_output.md` containing the final blog article.

## Agent Pipeline Flow

1. **Planner Agent** → Researches topic and creates outline
2. **Writer Agent** → Drafts article based on plan
3. **Editor Agent** → Refines and polishes content
4. **Output** → Publication-ready blog article

## Troubleshooting

### Before Deployment
- Run tests first: `python run_tests.py`
- Ensure virtual environment is active: `source venv/bin/activate`
- Check Python version: `python --version` (requires 3.9+)
- Verify packages: `pip list | grep crewai`
- Check API keys are set: `echo $OPENAI_API_KEY`

### During Execution
- Monitor execution: Add `verbose=True` in agent configuration
- Check logs: Application outputs detailed progress
- Resource usage: Run `monitor_resources.py` for tracking

### Common Issues
- **No LLM provider**: Set OPENAI_API_KEY or install Ollama
- **Import errors**: Reinstall dependencies with `pip install -r requirements.txt`
- **Permission errors**: Ensure test scripts are executable: `chmod +x *.sh`
- **Memory issues**: Close other applications or use lighter models
