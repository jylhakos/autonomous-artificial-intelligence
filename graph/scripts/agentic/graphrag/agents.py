from langgraph.prebuilt import create_react_agent

# Pack both capabilities into the toolset
tools = [search_unstructured_text, query_graph_relationships]

# Explicit system instructions routing instructions for Llama 3.2
system_instructions = (
    "You are a local Neo4j Knowledge Assistant powered by Llama 3.2.\n"
    "You have access to two tools:\n"
    "1. `search_unstructured_text`: Use this if the user asks for concepts, ideas, details or summaries of the text contents.\n"
    "2. `query_graph_relationships`: Use this if the user asks structural questions about network paths, node counts, or entity relations.\n"
    "Always rely on the tools to answer questions instead of guessing."
)

# Compile the LangGraph engine
agent_executor = create_react_agent(llm, tools, state_modifier=system_instructions)
