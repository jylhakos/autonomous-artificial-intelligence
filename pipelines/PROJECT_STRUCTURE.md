# Project Structure

This document visualizes the structure of the Vibe Coding with Foundry Local project.

## Directory Tree

```
pipelines/
├── 📄 README.md                          #  SDLC + Vibe Coding documentation
├── 📄 FOUNDRY_LOCAL_SETUP.md             # Step-by-step Linux setup guide
├── 📄 VENV_SETUP.md                      # Virtual environment setup documentation
├── 📄 PROJECT_STRUCTURE.md               # This file - project visualization
├── 📄 .gitignore                         # Git ignore patterns for Python projects
├── 📄 requirements.txt                   # Python package dependencies
│
├── 📁 scripts/                           # Executable scripts directory
│   ├── 📄 README.md                      # Scripts documentation and usage guide
│   ├── 🐍 prompt_assistant.py            # ⭐ Main vibe coding prompt assistant
│   ├── 🐍 check_venv.py                  # Virtual environment verification (Python)
│   ├── 🐍 example_script.py              # Example demonstrating venv usage
│   └── 🔧 check_venv.sh                  # Virtual environment verification (Bash)
│
├── 📁 venv/                              # Python virtual environment (ignored by git)
│   ├── bin/                              # Executables and activation scripts
│   ├── lib/                              # Installed Python packages
│   └── include/                          # Header files
│
└── 📁 .vscode/                           # VS Code workspace configuration
    ├── settings.json                     # Editor and Python settings
    └── launch.json                       # Debug configurations
```

## File Descriptions

### Documentation Files

| Icon | File                     | Purpose                                                                                                                                                                               | Lines  |
| ---- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 📄   | `README.md`              | Documentation covering SDLC, Vibe Coding methodology, AI agents, practical tutorials, Foundry Local integration, and AGENTS.md format. Includes 22 major sections with 75 references. | ~2,238 |
| 📄   | `FOUNDRY_LOCAL_SETUP.md` | Step-by-step guide for setting up vibe coding environment with VS Code, terminal, and Foundry Local on Linux platforms.                                                               | ~400   |
| 📄   | `VENV_SETUP.md`          | Virtual environment setup and best practices guide                                                                                                                                    | ~150   |
| 📄   | `PROJECT_STRUCTURE.md`   | This document - project visualization with architecture diagrams and usage patterns                                                                                                   | ~600   |

### Scripts Directory

| Icon | Directory/File                | Purpose                                                                                                                                                                                                                             | Technology                                 |
| ---- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| 📁   | `scripts/`                    | Organized directory containing all executable Python and shell scripts                                                                                                                                                              | -                                          |
| 📄   | `scripts/README.md`           | Scripts documentation with step-by-step instructions, troubleshooting, and setup guide links                                                                                                                                        | ~550 lines                                 |
| 🐍⭐ | `scripts/prompt_assistant.py` | **Main Application**: Interactive tool for generating optimized vibe coding prompts using Foundry Local's locally-running AI models. Features intent analysis, template generation, prompt refinement, and conversational workflow. | Python 3.12, OpenAI SDK, Foundry Local API |
| 🐍   | `scripts/check_venv.py`       | Virtual environment verification utility with detailed status reporting                                                                                                                                                             | Python 3.12                                |
| 🐍   | `scripts/example_script.py`   | Example demonstrating HTTP requests with venv and external library usage                                                                                                                                                            | Python 3.12, requests                      |
| 🔧   | `scripts/check_venv.sh`       | Bash script for verifying virtual environment activation status                                                                                                                                                                     | Bash                                       |

### Configuration Files

| Icon | File               | Purpose                                                        |
| ---- | ------------------ | -------------------------------------------------------------- |
| 📝   | `requirements.txt` | Python package dependencies with pinned versions (19 packages) |
| 🚫   | `.gitignore`       | Excludes venv/, **pycache**/, build artifacts, IDE configs     |

### VS Code Configuration

| Icon | Directory/File          | Purpose                                               |
| ---- | ----------------------- | ----------------------------------------------------- |
| 📁   | `.vscode/`              | VS Code workspace settings directory                  |
| ⚙️   | `.vscode/settings.json` | Python interpreter, linting, formatting configuration |
| 🐛   | `.vscode/launch.json`   | Debug configurations for prompt_assistant.py          |

### Virtual Environment

| Icon | Directory   | Purpose                                               |
| ---- | ----------- | ----------------------------------------------------- |
| 📁   | `venv/`     | Isolated Python environment (excluded from git)       |
| 📦   | `venv/bin/` | Python executable, pip, and activation scripts        |
| 📚   | `venv/lib/` | Installed packages (openai, requests, pydantic, etc.) |

## Technology Stack

### Core Technologies

```
┌─────────────────────────────────────────┐
│      Vibe Coding Environment            │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐   │
│  │   Visual Studio Code             │   │
│  │   + Foundry Toolkit Extension    │   │
│  │   + Python Extension             │   │
│  │   + GitHub Copilot (Optional)    │   │
│  └──────────────────────────────────┘   │
│                 ↕                       │
│  ┌──────────────────────────────────┐   │
│  │   Python 3.12.3 Application      │   │
│  │   (prompt_assistant.py)          │   │
│  └──────────────────────────────────┘   │
│                 ↕                       │
│  ┌──────────────────────────────────┐   │
│  │   OpenAI-Compatible API Client   │   │
│  │   (openai==2.32.0)               │   │
│  └──────────────────────────────────┘   │
│                 ↕                       │
│  ┌──────────────────────────────────┐   │
│  │   Microsoft Foundry Local        │   │
│  │   http://localhost:8080          │   │
│  └──────────────────────────────────┘   │
│                 ↕                       │
│  ┌──────────────────────────────────┐   │
│  │   Local AI Models                │   │
│  │   • qwen2.5-0.5b                 │   │
│  │   • phi-3-mini                   │   │
│  │   • llama-3.2-1b                 │   │
│  └──────────────────────────────────┘   │
│                 ↕                       │
│  ┌──────────────────────────────────┐   │
│  │   Hardware Acceleration          │   │
│  │   NPU / GPU / CPU                │   │
│  └──────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

### Python Dependencies

**Core Packages:**

- `openai==2.32.0` - OpenAI-compatible API client for Foundry Local
- `requests==2.33.1` - HTTP library for API connectivity checks
- `pydantic==2.13.3` - Data validation and settings management
- `httpx==0.28.1` - Modern HTTP client (used by openai)

**Supporting Packages:**

- `anyio==4.13.0` - Async I/O support
- `tqdm==4.67.3` - Progress bars
- `typing-extensions==4.15.0` - Type hint extensions

**Dependencies:** 19 packages (see `requirements.txt`)

## Application Architecture

### Prompt Assistant Components

```
prompt_assistant.py (in scripts/)
├── 📦 PromptCategory (Enum)
│   └── Defines prompt types: code_generation, refactoring, debugging,
│       testing, documentation, architecture, data_pipeline, ui_design
│
├── 📦 PromptTemplate (Dataclass)
│   ├── category: str
│   ├── intent: str
│   ├── context: str
│   ├── constraints: List[str]
│   ├── expected_output: str
│   └── to_prompt() → str
│
├── 📦 FoundryLocalClient (Class)
│   ├── __init__(base_url, model)
│   ├── generate_completion(prompt, system_prompt, temperature, max_tokens) → str
│   └── generate_with_tools(prompt, tools, system_prompt) → Dict
│
├── 📦 PromptAssistant (Class)
│   ├── analyze_intent(user_request) → Dict[str, Any]
│   ├── generate_prompt_template(analysis) → PromptTemplate
│   ├── refine_prompt(template, refinement_request) → PromptTemplate
│   ├── generate_vibe_coding_prompt(user_request) → str
│   └── interactive_session() → None
│
└── 🎯 main()
    ├── check_foundry_local_status() → bool
    ├── Initialize FoundryLocalClient
    ├── Create PromptAssistant
    └── Run interactive or CLI mode
```

## Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Vibe Coding Workflow                          │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  User launches  │
                  │ prompt_assistant│
                  └────────┬────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Describes what to     │
              │  build in natural      │
              │  language              │
              └───────────┬────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │  Foundry Local analyzes intent     │
         │  • Detects category                │
         │  • Extracts requirements           │
         │  • Identifies constraints          │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────────┐
         │  Generates optimized vibe coding    │
         │  prompt with:                       │
         │  • Clear goal statement             │
         │  • Sufficient context               │
         │  • Technical constraints            │
         │  • Expected output format           │
         └────────────┬────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ User reviews  │
              │ generated     │
              │ prompt        │
              └───────┬───────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
  ┌──────────────┐      ┌──────────────────┐
  │ Accept and   │      │ Type 'refine'    │
  │ use in VS    │      │ to improve       │
  │ Code/Copilot │      │ prompt           │
  └──────────────┘      └────────┬─────────┘
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │ Provides refinement   │
                    │ instructions          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Foundry Local refines │
                    │ and regenerates       │
                    └───────────┬───────────┘
                                │
                                └──────────┐
                                          │
                                          ▼
                              ┌──────────────────┐
                              │ Iterates until   │
                              │ prompt is        │
                              │ optimal          │
                              └──────────────────┘
```

## Usage Patterns

### Pattern 1: Interactive Prompt Generation

```bash
# Terminal
$ source venv/bin/activate
$ python scripts/prompt_assistant.py

# Interactive session starts
💬 Your request: Create a FastAPI REST API with PostgreSQL database for blog posts
🔍 Analyzing your request with Foundry Local...
✓ Detected category: code_generation
✓ Intent: Build REST API for blog management
📝 Generating optimized vibe coding prompt...
✨ GENERATED VIBE CODING PROMPT:
[... optimized prompt output ...]
```

### Pattern 2: Command-Line Mode

```bash
$ python scripts/prompt_assistant.py "Create a data pipeline that processes CSV files and generates visualizations"

# Single request, immediate output
```

### Pattern 3: VS Code Integration

1. Open integrated terminal in VS Code
2. Run prompt assistant interactively
3. Copy generated prompt
4. Paste into GitHub Copilot Chat or code editor
5. Let AI implement the code
6. Test and refine conversationally

## Key Features

### Prompt Assistant Capabilities

**Intent Analysis**

- Automatically detects what type of code user wants to create
- Extracts requirements and constraints from natural language
- Categorizes requests (code generation, refactoring, debugging, etc.)

  **Smart Prompt Generation**

- Creates structured, detailed prompts for AI coding assistants
- Includes context, constraints, and expected outputs
- Follows vibe coding best practices

  **Interactive Refinement**

- Allows iterative improvement of generated prompts
- Conversational feedback loop
- Maintains context across refinements

  **Local Processing**

- All AI processing happens on-device via Foundry Local
- No data sent to cloud
- Privacy and offline capability

  **OpenAI Compatible**

- Drop-in replacement for OpenAI API
- Works with existing OpenAI SDK code
- Easy integration with other tools

### Hardware Optimization

The system leverages available hardware automatically:

```
Priority Order:
1. NPU (Neural Processing Unit)  ← Fastest, lowest power
2. GPU (Graphics Processing Unit) ← High performance
3. CPU (Central Processing Unit)  ← Universal fallback
```

Foundry Local automatically selects the best available hardware for optimal performance.

## Development Workflow

### For Developers

1. **Setup:** Follow `FOUNDRY_LOCAL_SETUP.md`
2. **Activate:** `source venv/bin/activate`
3. **Develop:** Edit scripts in `scripts/` directory or create new ones
4. **Test:** Run with `python scripts/prompt_assistant.py`
5. **Debug:** Use VS Code debugger (F5)
6. **Deploy:** Share with team or integrate into CI/CD

### For End Users

1. **Activate:** `source venv/bin/activate`
2. **Run:** `python scripts/prompt_assistant.py`
3. **Request:** Describe what you want to build
4. **Copy:** Use generated prompt in VS Code/Copilot
5. **Build:** Let AI generate the code
6. **Verify:** Test and refine iteratively

## Extension Opportunities

### Potential Enhancements

1. **Web Interface:** Create Flask/FastAPI web UI for prompt assistant
2. **Prompt Library:** Save and share successful prompts
3. **Team Collaboration:** Multi-user prompt refinement
4. **Model Selection:** Dynamic model switching based on task
5. **Code Execution:** Direct code generation and execution
6. **Integration:** Connect to GitHub Copilot API
7. **Analytics:** Track prompt effectiveness metrics
8. **Templates:** Pre-built templates for common tasks

### Customization Points

- **Models:** Try different Foundry Local models (phi-3, llama, etc.)
- **Categories:** Add custom prompt categories
- **Templates:** Create domain-specific prompt templates
- **Tools:** Implement additional tool calling functions
- **Workflows:** Build custom vibe coding workflows

## Performance Considerations

### Model Selection Guide

| Model        | Size | Speed  | Quality  | Use Case                          |
| ------------ | ---- | ------ | -------- | --------------------------------- |
| qwen2.5-0.5b | 0.5B | ⚡⚡⚡ | ⭐⭐     | Quick prototyping, simple prompts |
| qwen2.5-1.5b | 1.5B | ⚡⚡   | ⭐⭐⭐   | Balanced performance              |
| phi-3-mini   | 3.8B | ⚡     | ⭐⭐⭐⭐ | Complex prompts, high quality     |
| llama-3.2-1b | 1B   | ⚡⚡   | ⭐⭐⭐   | General purpose                   |

### Resource Requirements

**Minimum Configuration:**

- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB
- Model: qwen2.5-0.5b

**Recommended Configuration:**

- CPU: 8+ cores or GPU
- RAM: 16GB+
- Storage: 20GB+
- Model: phi-3-mini or larger

**Optimal Configuration:**

- NPU: Intel/AMD NPU
- RAM: 32GB+
- Storage: SSD 50GB+
- Model: Multiple models for different tasks

## Security and Privacy

### Data Flow

```
User Input → prompt_assistant.py → Foundry Local (localhost:8080) → AI Model (on-device) → Response
     ↑                                                                                          │
     └──────────────────────────────────────────────────────────────────────────────────────────┘
                                ALL DATA STAYS LOCAL
```

### Security Features

**Local Processing:** No cloud API calls
**No API Keys:** No credentials needed for local operation
**Offline Capable:** Works without internet connection
**Data Privacy:** All data remains on your device
**Open Source:** Transparent, auditable code

### Best Practices

1. **Review Generated Code:** Always inspect AI outputs before deployment
2. **Secure Endpoints:** If exposing Foundry Local, use authentication
3. **Model Provenance:** Only use models from trusted sources
4. **Version Control:** Track prompt templates in git
5. **Access Control:** Restrict who can run the assistant in production

---

**Last Updated:** April 23, 2026

**License:** MIT
