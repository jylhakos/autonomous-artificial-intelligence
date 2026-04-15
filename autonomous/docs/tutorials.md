# Tutorials

## Tutorial 1: Building Your First AI Agent

### Objective

Create a simple autonomous agent that can perform tasks using tools.

### Steps

#### Step 1: Import Required Modules

```python
from agents.base_agent import SimpleAgent
from tools.calculator_tool import CalculatorTool
```

#### Step 2: Create an Agent

```python
# Initialize the agent
agent = SimpleAgent(name="MyFirstAgent")

# Add tools
calculator = CalculatorTool()
agent.add_tool(calculator)
```

#### Step 3: Define a Goal

```python
goal = "Calculate the sum of 25 and 75"
```

#### Step 4: Execute

```python
result = agent.run(goal)
print(f"Result: {result}")
```

### Complete Example

```python
from agents.base_agent import SimpleAgent
from tools.calculator_tool import CalculatorTool

def main():
    # Create agent
    agent = SimpleAgent("MathBot")

    # Add calculator tool
    calculator = CalculatorTool()
    agent.add_tool(calculator)

    # Execute goal
    goal = "Calculate the sum of 25 and 75"
    result = agent.run(goal)

    # Use tool directly
    calc_result = agent.use_tool(
        "calculator",
        operation="add",
        a=25,
        b=75
    )
    print(f"Calculator result: {calc_result}")

if __name__ == "__main__":
    main()
```

---

## Tutorial 2: Creating a Multi-Agent System

### Objective

Build a supervisor-based multi-agent system with specialized agents.

### Architecture

```
Supervisor Agent
    ├── Research Agent (Information gathering)
    ├── Code Agent (Code generation)
    └── Analysis Agent (Data analysis)
```

### Steps

#### Step 1: Define Specialized Agents

```python
from agents.specialized_agents import (
    ResearchAgent,
    CodeAgent,
    AnalysisAgent
)

# Create specialized agents
research_agent = ResearchAgent()
code_agent = CodeAgent()
analysis_agent = AnalysisAgent()
```

#### Step 2: Create Supervisor

```python
from agents.supervisor_agent import SupervisorAgent

supervisor = SupervisorAgent()

# Register agents
supervisor.register_agent(research_agent)
supervisor.register_agent(code_agent)
supervisor.register_agent(analysis_agent)
```

#### Step 3: Execute Workflow

```python
goal = "Analyze user engagement metrics and create a dashboard"

result = supervisor.execute_workflow(goal)
```

### Complete Example

See `examples/multi_agent_system.py`

---

## Tutorial 3: Integrating with LiteLLM

### Objective

Use LiteLLM to interact with multiple language models.

### Steps

#### Step 1: Install LiteLLM

```bash
pip install litellm
```

#### Step 2: Configure API Keys

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

#### Step 3: Create Multi-Model Interface

```python
from litellm import completion

def chat_with_model(model_name, message):
    response = completion(
        model=model_name,
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# Try different models
models = ["gpt-4", "claude-3-sonnet-20240229", "gemini-pro"]

for model in models:
    response = chat_with_model(model, "Explain AI agents in one sentence")
    print(f"{model}: {response}\n")
```

### Complete Example

See `examples/multi_model_chat.py`

---

## Tutorial 4: Adding Memory to Agents

### Objective

Implement short-term and long-term memory for agents.

### Steps

#### Step 1: Create Memory System

```python
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

class MemoryAgent:
    def __init__(self, name):
        self.name = name
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def remember(self, item, memory_type="short"):
        if memory_type == "short":
            self.short_term.add(item)
        else:
            self.long_term.add(item)

    def recall(self, query, memory_type="short"):
        if memory_type == "short":
            return self.short_term.retrieve(query)
        else:
            return self.long_term.retrieve(query)
```

#### Step 2: Use Memory in Workflow

```python
agent = MemoryAgent("MemBot")

# Store information
agent.remember({
    "task": "Process user data",
    "result": "Completed successfully",
    "timestamp": "2026-04-14"
})

# Retrieve information
past_tasks = agent.recall("previous tasks")
```

---

## Tutorial 5: Implementing Observability

### Objective

Add tracing and monitoring to your agents.

### Steps

#### Step 1: Setup Tracing

```python
from observability.tracing import AgentTracer

tracer = AgentTracer()

# Start trace
trace_id = tracer.start_trace("workflow_execution")

# Log events
tracer.log_event(trace_id, "agent_started", {
    "agent_name": "ResearchBot",
    "task": "Search documentation"
})

# End trace
tracer.end_trace(trace_id)
```

#### Step 2: Setup Metrics

```python
from observability.metrics import MetricsCollector

metrics = MetricsCollector()

# Record metrics
metrics.record("task_duration", 1.5, {"task_type": "research"})
metrics.record("tool_invocations", 3, {"tool": "calculator"})

# Get metrics
stats = metrics.get_stats()
print(f"Average task duration: {stats['task_duration']['avg']}")
```

#### Step 3: Integrate with Langfuse

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create trace
trace = langfuse.trace(name="agent-workflow")

# Add span
span = trace.span(name="research-phase")
span.end(output={"findings": "..."})
```

---

## Tutorial 6: Building a DevOps Agent

### Objective

Create an agent that can help with DevOps tasks.

### Capabilities

- Monitor application logs
- Detect anomalies
- Suggest fixes
- Execute remediation actions

### Steps

#### Step 1: Define Tools

```python
class LogAnalysisTool:
    def analyze_logs(self, log_file):
        # Analyze logs for errors
        pass

class MetricsTool:
    def get_metrics(self, service_name):
        # Fetch service metrics
        pass

class RemediationTool:
    def restart_service(self, service_name):
        # Restart a service
        pass
```

#### Step 2: Create DevOps Agent

```python
class DevOpsAgent:
    def __init__(self):
        self.log_analyzer = LogAnalysisTool()
        self.metrics = MetricsTool()
        self.remediation = RemediationTool()

    def monitor_and_fix(self, service_name):
        # Get metrics
        metrics = self.metrics.get_metrics(service_name)

        # Check for issues
        if metrics['error_rate'] > 0.05:
            # Analyze logs
            issues = self.log_analyzer.analyze_logs(
                f"/var/log/{service_name}.log"
            )

            # Suggest fix
            if issues['severity'] == 'high':
                self.remediation.restart_service(service_name)
```

---

## Next Steps

1. Explore the [Architecture Guide](architecture.md)
2. Review example code in `examples/`

## Resources

- [Main README](../README.md)
- [Setup Guide](setup.md)
- [VS Code Agents Documentation](https://code.visualstudio.com/docs/copilot/agents/overview)
