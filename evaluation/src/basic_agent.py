"""
Basic AI Agent Example with LangChain

This module demonstrates how to create a simple AI agent using LangChain
with tool integration and basic functionality.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Annotated
import os


@tool
def get_weather(city: Annotated[str, "The name of the city"]) -> str:
    """Get weather information for a given city."""
    # This is a mock function - in production, you'd call a real weather API
    weather_data = {
        "san francisco": "It's always sunny in San Francisco! Temperature: 72°F",
        "new york": "Partly cloudy in New York. Temperature: 65°F",
        "london": "Rainy in London. Temperature: 55°F",
        "tokyo": "Clear skies in Tokyo. Temperature: 68°F",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(
    operation: Annotated[str, "The operation to perform: add, subtract, multiply, divide"],
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> str:
    """Perform basic mathematical calculations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
    }
    
    if operation.lower() in operations:
        result = operations[operation.lower()](a, b)
        return f"The result of {operation} {a} and {b} is: {result}"
    else:
        return f"Unknown operation: {operation}"


def create_basic_agent():
    """
    Create a basic AI agent with tool-calling capabilities.
    
    Returns:
        AgentExecutor: Configured agent executor ready to process requests
    """
    # Initialize the language model
    # You can also use ChatOpenAI or other supported models
    llm = ChatAnthropic(
        model="claude-sonnet-4",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    
    # Define available tools
    tools = [get_weather, calculate]
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools. "
                   "Use the tools when necessary to answer user questions accurately."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent_executor


def main():
    """Main function to demonstrate basic agent usage."""
    print("=" * 60)
    print("Basic AI Agent with LangChain")
    print("=" * 60)
    
    # Create the agent
    agent = create_basic_agent()
    
    # Example queries
    queries = [
        "What is the weather in San Francisco?",
        "Calculate 15 multiplied by 8",
        "What's the weather in Tokyo and what is 100 divided by 4?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 60)
        
        try:
            response = agent.invoke({"input": query})
            print(f"[Response]: {response['output']}")
        except Exception as e:
            print(f"[Error]: {str(e)}")
        
        print("-" * 60)


if __name__ == "__main__":
    main()
