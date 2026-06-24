response_graph = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "How many total DocumentChunk nodes currently exist in my database graph?"}]}
)
print("\n--- Cypher Graph Output ---")
print(response_graph["messages"][-1].content)
