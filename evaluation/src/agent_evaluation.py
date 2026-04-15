"""
AI Agent Evaluation with LangSmith

This module demonstrates how to evaluate AI agents using LangSmith's
evaluation framework, including dataset creation, custom evaluators,
and running experiments.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from typing import Annotated, Dict, Any
import os


# Initialize LangSmith client
client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))


@tool
def get_product_price(product_name: Annotated[str, "The name of the product"]) -> str:
    """Get the price of a product."""
    prices = {
        "laptop": "$999",
        "phone": "$699",
        "tablet": "$499",
        "headphones": "$199",
        "mouse": "$49",
    }
    return prices.get(product_name.lower(), "Product not found")


@tool
def check_inventory(product_name: Annotated[str, "The name of the product"]) -> str:
    """Check if a product is in stock."""
    inventory = {
        "laptop": "In stock (15 units)",
        "phone": "In stock (32 units)",
        "tablet": "Low stock (3 units)",
        "headphones": "Out of stock",
        "mouse": "In stock (50 units)",
    }
    return inventory.get(product_name.lower(), "Product not found")


def create_eval_agent():
    """Create an agent for evaluation."""
    llm = ChatAnthropic(
        model="claude-sonnet-4",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    
    tools = [get_product_price, check_inventory]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful shopping assistant. "
         "Help users find product information including prices and availability. "
         "Always check both price and inventory when asked about a product."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    
    return agent_executor


def create_evaluation_dataset():
    """
    Create a golden dataset for agent evaluation.
    
    A golden dataset consists of input-output pairs representing
    expected agent behavior for various scenarios.
    
    Returns:
        str: Name of the created dataset
    """
    dataset_name = "agent-evaluation-dataset"
    
    # Define evaluation examples with inputs and expected outputs
    examples = [
        {
            "input": "How much does a laptop cost?",
            "expected_output": "laptop price should be $999",
            "expected_tools": ["get_product_price"],
        },
        {
            "input": "Is the tablet available?",
            "expected_output": "tablet availability information",
            "expected_tools": ["check_inventory"],
        },
        {
            "input": "I want to buy a phone. Can you tell me the price and if it's in stock?",
            "expected_output": "phone costs $699 and is in stock",
            "expected_tools": ["get_product_price", "check_inventory"],
        },
        {
            "input": "What about headphones?",
            "expected_output": "headphones cost $199 but are out of stock",
            "expected_tools": ["get_product_price", "check_inventory"],
        },
        {
            "input": "Give me details on the mouse",
            "expected_output": "mouse costs $49 and is in stock",
            "expected_tools": ["get_product_price", "check_inventory"],
        },
    ]
    
    # Check if dataset exists, if not create it
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists")
    except Exception:
        # Create new dataset
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden dataset for evaluating shopping assistant agent",
        )
        
        # Add examples to dataset
        for example in examples:
            client.create_example(
                inputs={"input": example["input"]},
                outputs={
                    "expected_output": example["expected_output"],
                    "expected_tools": example["expected_tools"],
                },
                dataset_id=dataset.id,
            )
        
        print(f"Created dataset '{dataset_name}' with {len(examples)} examples")
    
    return dataset_name


# Custom Evaluators

def accuracy_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Evaluate if the agent's final answer is accurate.
    
    This is a simple keyword-based evaluator. In production,
    you might use an LLM-as-a-judge for more sophisticated evaluation.
    """
    agent_output = run.outputs.get("output", "").lower()
    expected_keywords = example.outputs.get("expected_output", "").lower()
    
    # Simple keyword matching
    score = 1.0 if any(keyword in agent_output for keyword in expected_keywords.split()) else 0.0
    
    return {
        "key": "accuracy",
        "score": score,
        "comment": f"Output contains expected keywords" if score > 0 else "Output missing expected keywords",
    }


def tool_usage_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Evaluate if the agent used the correct tools.
    
    This checks the trajectory - did the agent call the right tools
    to answer the question?
    """
    expected_tools = set(example.outputs.get("expected_tools", []))
    
    # Extract tools used from intermediate steps
    used_tools = set()
    if "intermediate_steps" in run.outputs:
        for action, _ in run.outputs["intermediate_steps"]:
            used_tools.add(action.tool)
    
    # Check if all expected tools were used
    correct_tools = expected_tools.issubset(used_tools)
    score = 1.0 if correct_tools else 0.5 if len(used_tools & expected_tools) > 0 else 0.0
    
    return {
        "key": "tool_usage",
        "score": score,
        "comment": f"Expected: {expected_tools}, Used: {used_tools}",
    }


def efficiency_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Evaluate agent efficiency - did it use minimal tool calls?
    
    Penalize agents that make unnecessary tool calls.
    """
    expected_tools_count = len(example.outputs.get("expected_tools", []))
    
    actual_tools_count = 0
    if "intermediate_steps" in run.outputs:
        actual_tools_count = len(run.outputs["intermediate_steps"])
    
    # Score based on efficiency
    if actual_tools_count == 0:
        score = 0.0
    elif actual_tools_count == expected_tools_count:
        score = 1.0
    elif actual_tools_count < expected_tools_count:
        score = 0.5  # Agent skipped necessary tools
    else:
        # Penalize extra tool calls
        score = max(0.0, 1.0 - (actual_tools_count - expected_tools_count) * 0.2)
    
    return {
        "key": "efficiency",
        "score": score,
        "comment": f"Expected {expected_tools_count} tools, used {actual_tools_count}",
    }


def llm_as_judge_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Use an LLM to evaluate the quality of the agent's response.
    
    This is a more sophisticated evaluator that uses another LLM
    to judge the agent's output quality.
    """
    judge_llm = ChatAnthropic(
        model="claude-sonnet-4",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    
    agent_output = run.outputs.get("output", "")
    user_query = example.inputs.get("input", "")
    expected_output = example.outputs.get("expected_output", "")
    
    judge_prompt = f"""
    You are evaluating an AI agent's response to a user query.
    
    User Query: {user_query}
    Agent Response: {agent_output}
    Expected Information: {expected_output}
    
    Evaluate the agent's response on a scale of 0-1 based on:
    - Accuracy: Does it provide correct information?
    - Completeness: Does it address all aspects of the query?
    - Clarity: Is the response clear and well-formatted?
    
    Respond with ONLY a number between 0 and 1.
    """
    
    try:
        result = judge_llm.invoke(judge_prompt)
        score = float(result.content.strip())
        score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
    except Exception:
        score = 0.5  # Default score if evaluation fails
    
    return {
        "key": "llm_judge_quality",
        "score": score,
        "comment": "LLM-as-judge evaluation",
    }


def run_evaluation():
    """
    Run a complete evaluation experiment on the agent.
    
    This will:
    1. Load the golden dataset
    2. Run the agent on each example
    3. Apply all evaluators
    4. Generate a report with scores
    """
    print("=" * 60)
    print("AI Agent Evaluation with LangSmith")
    print("=" * 60)
    
    # Create or load dataset
    dataset_name = create_evaluation_dataset()
    
    # Create the agent to evaluate
    agent = create_eval_agent()
    
    # Define the target function for evaluation
    def agent_target(inputs: dict) -> dict:
        """Wrapper function for the agent."""
        result = agent.invoke({"input": inputs["input"]})
        return {"output": result["output"], "intermediate_steps": result.get("intermediate_steps", [])}
    
    # Run the evaluation
    print(f"\nRunning evaluation on dataset: {dataset_name}")
    print("Evaluators: accuracy, tool_usage, efficiency, llm_as_judge")
    print("-" * 60)
    
    try:
        results = evaluate(
            agent_target,
            data=dataset_name,
            evaluators=[
                accuracy_evaluator,
                tool_usage_evaluator,
                efficiency_evaluator,
                # llm_as_judge_evaluator,  # Uncomment to enable LLM-as-judge
            ],
            experiment_prefix="agent-eval",
            description="Evaluation of shopping assistant agent",
        )
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        print(f"View results at: https://smith.langchain.com/")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")


def compare_agent_versions():
    """
    Compare different agent configurations or prompts.
    
    This demonstrates how to run A/B testing on agents by
    evaluating multiple versions on the same dataset.
    """
    print("\n" + "=" * 60)
    print("Comparing Agent Configurations")
    print("=" * 60)
    
    # You can create multiple agent versions with different:
    # - Prompts
    # - Models
    # - Tools
    # - Parameters (temperature, etc.)
    
    # Then run evaluate() on each version and compare results
    print("To compare versions, run evaluate() multiple times with different agent configurations")
    print("LangSmith will automatically track and compare the results")


def main():
    """Main function to run agent evaluation."""
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("ERROR: LANGCHAIN_API_KEY not set")
        print("Please set your LangSmith API key in the .env file")
        return
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Please set your Anthropic API key in the .env file")
        return
    
    # Run the evaluation
    run_evaluation()
    
    # Show how to compare versions
    compare_agent_versions()


if __name__ == "__main__":
    main()
