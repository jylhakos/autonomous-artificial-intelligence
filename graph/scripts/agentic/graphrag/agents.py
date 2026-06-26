from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from tools import search_unstructured_text, query_graph_relationships

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")

# Pack both capabilities into the toolset
tools = [search_unstructured_text, query_graph_relationships]

# Explicit system instructions routing instructions for Llama 3.2
system_instructions = (
    "You are a local Neo4j Knowledge Assistant powered by Llama 3.2 specialized in the Iris dataset.\n"
    "You have access to two tools:\n"
    "1. `search_unstructured_text`: Use this if the user asks for concepts, morphological characteristics, "
    "species descriptions, or summaries of the Iris dataset text contents.\n"
    "2. `query_graph_relationships`: Use this if the user asks structural questions about relationships "
    "between species, specimens, and measurements, node counts, or entity relations in the graph database.\n"
    "Always rely on the tools to answer questions instead of guessing."
)

# Compile the LangGraph engine
agent_executor = create_react_agent(llm, tools, state_modifier=system_instructions)
