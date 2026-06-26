from agents import agent_executor

# Test vector search with a semantic query about the Iris dataset
response_vector = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "What are the main morphological characteristics of Iris virginica?"}]}
)
print("--- Vector Retrieval Output ---")
print(response_vector["messages"][-1].content)
