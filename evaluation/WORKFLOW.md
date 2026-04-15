# Agent Evaluation Workflow

This document describes the workflow and architecture of the agent evaluation framework.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                        │
│                           (main.py)                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─────────────────────────────────────────────┐
                     │                                             │
                     ▼                                             ▼
        ┌────────────────────────┐                   ┌────────────────────────┐
        │    Agent Creation      │                   │    Evaluation Layer    │
        │   (LangChain/Tools)    │                   │     (LangSmith)        │
        └────────┬───────────────┘                   └────────┬───────────────┘
                 │                                            │
                 ├──────────────┬──────────────┬──────────────┤
                 ▼              ▼              ▼              ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
         │ Tools    │   │ Prompts  │   │ Traces   │   │ Datasets │
         └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## Evaluation Workflow

### Phase 1: Development and Testing

```
1. Create Agent
   └─> Define tools and capabilities
   └─> Configure prompts and model
   └─> Set up tracing

2. Enable Observability
   └─> Configure LangSmith API keys
   └─> Enable automatic tracing
   └─> Set project name

3. Test Locally
   └─> Run basic examples
   └─> Verify tool calls work
   └─> Check traces in LangSmith
```

### Phase 2: Offline Evaluation

```
1. Create Golden Dataset
   └─> Collect representative examples
   └─> Define expected outputs
   └─> Specify expected tool calls

2. Define Evaluators
   ├─> Accuracy evaluator
   ├─> Tool usage evaluator
   ├─> Efficiency evaluator
   └─> LLM-as-judge evaluator

3. Run Experiments
   └─> Execute agent on dataset
   └─> Apply all evaluators
   └─> Generate scores and reports

4. Iterate and Improve
   └─> Analyze failures
   └─> Modify prompts/tools
   └─> Re-run evaluation
   └─> Compare versions
```

### Phase 3: Production Deployment

```
1. Deploy Agent
   └─> Enable production tracing
   └─> Set up monitoring

2. Online Evaluation
   ├─> Monitor latency metrics
   ├─> Track error rates
   ├─> Analyze tool usage patterns
   └─> Evaluate response quality

3. Continuous Improvement
   └─> Collect production traces
   └─> Add to golden dataset
   └─> Re-evaluate periodically
   └─> Deploy improvements
```

## Observability Primitives

### Run (Single-step Evaluation)

```
Input: "What's the weather in SF?"
   │
   └─> LLM Call
       └─> Tool Selection: get_weather
           └─> Tool Input: {city: "San Francisco"}
               └─> Tool Output: "Sunny, 72°F"
                   └─> Final Answer: "It's sunny and 72°F in San Francisco"

Evaluation:
✓ Correct tool selected
✓ Correct arguments
✓ Accurate final answer
```

### Trace (Full-turn Evaluation)

```
Input: "I need info on account ACC001"
   │
   ├─> Step 1: get_account_balance(ACC001)
   │   └─> Output: "$5,250.00"
   │
   ├─> Step 2: get_recent_transactions(ACC001)
   │   └─> Output: [list of transactions]
   │
   └─> Final Answer: "Account ACC001 has balance of $5,250.00.
                      Recent transactions: [...]"

Evaluation:
✓ Correct tool sequence
✓ All necessary information gathered
✓ Complete and accurate response
```

### Thread (Multi-turn Evaluation)

```
Turn 1:
User: "What's the balance for ACC001?"
Agent: "$5,250.00"

Turn 2:
User: "Transfer $100 to ACC002"
Agent: [Uses context from Turn 1, knows source account]
       "Transfer successful from ACC001 to ACC002"

Evaluation:
✓ Context maintained across turns
✓ Implicit reference resolved correctly
✓ Appropriate tool usage
```

## Evaluation Metrics

### Accuracy Metrics

- **Factual Correctness:** Is the information accurate?
- **Completeness:** Did the agent provide all necessary information?
- **Relevance:** Is the response aligned with the user's intent?

### Performance Metrics

- **Latency:** How long did execution take?
- **Token Usage:** How efficient is the agent with tokens?
- **Tool Call Count:** Did the agent make unnecessary calls?

### Quality Metrics

- **Hallucination Rate:** How often does the agent make up information?
- **Error Rate:** What percentage of requests fail?
- **Tool Accuracy:** Are tools called with correct arguments?

### User Experience Metrics

- **Conciseness:** Is the response appropriately brief?
- **Clarity:** Is the response easy to understand?
- **Helpfulness:** Does the response solve the user's problem?

## LLM-as-Judge Pattern

Using a high-capability model to evaluate agent outputs:

```python
Judge Prompt:
"
Evaluate this agent response:
User Query: {query}
Agent Response: {response}
Expected Info: {expected}

Rate 0-1 on:
- Accuracy
- Completeness
- Clarity
"

Judge Output: 0.85 → Agent scored 85%
```

## Comparison Workflow

Comparing different agent configurations:

```
Experiment 1: Claude Sonnet 4 + Temperature 0
Experiment 2: Claude Sonnet 4 + Temperature 0.5
Experiment 3: GPT-4 + Temperature 0

Run all three on same dataset
   │
   └─> Compare metrics:
       ├─> Accuracy: Exp1 (92%), Exp2 (88%), Exp3 (90%)
       ├─> Latency: Exp1 (2.1s), Exp2 (2.3s), Exp3 (3.1s)
       └─> Cost: Exp1 ($0.05), Exp2 ($0.05), Exp3 ($0.08)

Decision: Choose Experiment 1 (best accuracy + speed + cost)
```

## Monitoring Dashboard (Conceptual)

```
┌─────────────────────────────────────────────────────────────┐
│ Production Metrics (Last 24 Hours)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Total Requests: 1,247                                       │
│ Success Rate: 97.8%                                         │
│ Average Latency: 1.8s                                       │
│ Error Rate: 2.2%                                            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Tool Usage:                                                 │
│   get_weather: 453 calls                                    │
│   calculate: 287 calls                                      │
│   search_database: 507 calls                                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Alerts:                                                     │
│   ⚠️ High latency detected at 14:23 (spike to 5.2s)        │
│   ✓ All other metrics normal                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Best Practices

### 1. Start with Tracing

Always enable tracing from day one. Traces are your primary debugging tool.

### 2. Build Golden Datasets Incrementally

Start with 5-10 critical test cases. Add more as you discover edge cases in production.

### 3. Use Multiple Evaluators

Combine automated metrics (tool usage, latency) with LLM-as-judge for quality.

### 4. Monitor Production Continuously

Set up alerts for key metrics. Don't wait for users to report issues.

### 5. Iterate Based on Data

Use evaluation results to guide improvements. Don't guess what needs fixing.

### 6. Version Your Experiments

Name experiments descriptively (e.g., "claude-temp0-v1") to track changes over time.

### 7. Balance Speed and Quality

Fast responses are important, but accuracy matters more. Find the right tradeoff.

## Integration Points

### Where This Framework Fits

```
Your Application
   │
   ├─> Agent Logic (LangChain) ← This Framework Helps Here
   │   ├─> Tools
   │   ├─> Prompts
   │   └─> Model Config
   │
   ├─> Observability (LangSmith) ← This Framework Helps Here
   │   ├─> Tracing
   │   ├─> Datasets
   │   └─> Evaluation
   │
   └─> Business Logic
       └─> Your Domain-Specific Code
```

## Next Steps

1. Run the examples in this framework
2. Customize agents for your use case
3. Create your own golden datasets
4. Define domain-specific evaluators
5. Deploy with monitoring enabled
6. Continuously improve based on data

## References

- See [README.md](README.md) for full documentation
- See [SETUP.md](SETUP.md) for installation instructions
- Visit [LangSmith Docs](https://docs.langchain.com/langsmith/) for advanced features
