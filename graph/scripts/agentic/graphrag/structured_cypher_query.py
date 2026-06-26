from agents import agent_executor

# Test graph query to count nodes or explore relationships
response_graph = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "How many DocumentChunk nodes exist in the database and what species are represented?"}]}
)
print("\n--- Cypher Graph Output ---")
print(response_graph["messages"][-1].content)
