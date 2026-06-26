from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. Initialize ChromaDB connection
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
chroma_client = Chroma(
    collection_name="iris_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 2. Define Agent State
class GraphState(TypedDict):
    messages: List[BaseMessage]
    documents: Optional[List[str]]
    query: Optional[str]

# 3. Define Retriever Tool
@tool("retrieve_iris_documents")
def retrieve_iris_documents(query: str) -> List[str]:
    """Retrieve Iris dataset documents from local ChromaDB."""
    results = chroma_client.similarity_search(query, k=3)
    return [doc.page_content for doc in results]

tools = [retrieve_iris_documents]

# 3. Define Nodes
llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: GraphState):
    """The Agent node makes decisions on whether to use tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: GraphState):
    """Decides whether to route to tools or to finish."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 4. Assemble the LangGraph Workflow
workflow = StateGraph(GraphState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 5. Run the Agent
user_input = "What are the main characteristics of Iris setosa based on the morphological measurements?"
events = app.stream({"messages": [HumanMessage(content=user_input)]})
for event in events:
    print(event)
