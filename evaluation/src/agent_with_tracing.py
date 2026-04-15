"""
AI Agent with LangSmith Observability

This module demonstrates how to integrate LangSmith for tracing,
observability, and debugging of AI agent executions.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Annotated
import os


# Ensure LangSmith tracing is enabled
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "agent-evaluation-demo"
)


@tool
def search_database(
    query: Annotated[str, "The search query"],
    category: Annotated[str, "The category to search in"] = "general",
) -> str:
    """Search a mock database for information."""
    # Mock database
    database = {
        "products": {
            "laptop": "We have Dell XPS 13 and MacBook Air in stock",
            "phone": "iPhone 15 and Samsung Galaxy S24 available",
            "tablet": "iPad Air and Samsung Tab S9 in stock",
        },
        "general": {
            "hours": "We are open Monday-Friday 9AM-6PM",
            "location": "123 Main Street, Downtown",
            "contact": "Call us at 555-0123 or email info@example.com",
        },
    }
    
    category_data = database.get(category.lower(), database["general"])
    
    for key, value in category_data.items():
        if query.lower() in key.lower() or query.lower() in value.lower():
            return value
    
    return f"No results found for '{query}' in category '{category}'"


@tool
def get_user_info(user_id: Annotated[str, "The user ID to look up"]) -> str:
    """Retrieve user information from the system."""
    # Mock user database
    users = {
        "user123": "John Doe, Premium member since 2020, last order: 2024-04-01",
        "user456": "Jane Smith, Standard member since 2022, last order: 2024-03-15",
        "user789": "Bob Johnson, Premium member since 2019, last order: 2024-04-10",
    }
    
    return users.get(user_id, f"User {user_id} not found")


@tool
def process_order(
    user_id: Annotated[str, "The user ID"],
    product: Annotated[str, "The product name"],
    quantity: Annotated[int, "The quantity to order"],
) -> str:
    """Process a new order for a user."""
    # This is a mock function - in production, this would interface with an order system
    total_cost = quantity * 100  # Mock price calculation
    return (
        f"Order processed successfully!\n"
        f"User: {user_id}\n"
        f"Product: {product}\n"
        f"Quantity: {quantity}\n"
        f"Total: ${total_cost}"
    )


def create_traced_agent():
    """
    Create an AI agent with LangSmith tracing enabled.
    
    All agent executions will be automatically logged to LangSmith
    for observability, debugging, and evaluation.
    
    Returns:
        AgentExecutor: Configured agent executor with tracing
    """
    # Initialize the language model
    llm = ChatAnthropic(
        model="claude-sonnet-4",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    
    # Define available tools
    tools = [search_database, get_user_info, process_order]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful customer service assistant with access to various tools. "
         "Use the tools to help customers with their inquiries, orders, and information requests. "
         "Always be polite and provide accurate information based on the tool outputs."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the agent executor with metadata for tracing
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # Important for observability
    )
    
    return agent_executor


def run_traced_examples():
    """
    Run example queries with LangSmith tracing.
    
    Each execution will create a trace in LangSmith showing:
    - Input and output
    - Tool calls and their results (Thought-Action-Observation)
    - Token usage and latency
    - Error handling and retries
    """
    print("=" * 60)
    print("AI Agent with LangSmith Observability")
    print("=" * 60)
    print(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'agent-evaluation-demo')}")
    print(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2', 'enabled')}")
    print("=" * 60)
    
    # Create the traced agent
    agent = create_traced_agent()
    
    # Example queries that demonstrate different agent behaviors
    test_cases = [
        {
            "query": "What are your business hours?",
            "description": "Simple information retrieval",
        },
        {
            "query": "I want to order 2 laptops for user123",
            "description": "Multi-step task requiring multiple tool calls",
        },
        {
            "query": "What products do you have in the tablet category?",
            "description": "Category-specific search",
        },
        {
            "query": "Get me information about user456 and check if you have phones available",
            "description": "Multiple independent tool calls",
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}: {test_case['description']}")
        print(f"{'=' * 60}")
        print(f"Query: {test_case['query']}")
        print("-" * 60)
        
        try:
            # Run the agent - this will automatically create a trace in LangSmith
            result = agent.invoke({"input": test_case["query"]})
            
            print(f"\n[Agent Response]:")
            print(result["output"])
            
            # Display intermediate steps for observability
            if "intermediate_steps" in result:
                print(f"\n[Tool Calls Made]: {len(result['intermediate_steps'])}")
                for step_num, (action, observation) in enumerate(result["intermediate_steps"], 1):
                    print(f"  Step {step_num}: {action.tool} - {action.tool_input}")
            
        except Exception as e:
            print(f"\n[Error]: {str(e)}")
        
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("All traces have been sent to LangSmith!")
    print("View them at: https://smith.langchain.com/")
    print("=" * 60)


def main():
    """Main function to run traced agent examples."""
    # Verify API key is set
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("WARNING: LANGCHAIN_API_KEY not set. Tracing will not work.")
        print("Please set your LangSmith API key in the .env file")
        return
    
    run_traced_examples()


if __name__ == "__main__":
    main()
