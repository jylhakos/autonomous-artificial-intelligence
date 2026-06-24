response_vector = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "According to document.txt, what is the main theme?"}]}
)
print("--- Vector Retrieval Output ---")
print(response_vector["messages"][-1].content)
