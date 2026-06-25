from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. Define Agent State
class GraphState(TypedDict):
    messages: List[BaseMessage]
    documents: Optional[List[str]]
    query: Optional[str]

# 2. Define Retriever Tool
@tool("retrieve_documents")
def retrieve_documents(query: str) -> List[str]:
    """Retrieve documents from local ChromaDB."""
    # Assuming connection defined identically to the ingestion step
    results = chroma_client.similarity_search(query, k=3)
    return [doc.page_content for doc in results]

tools = [retrieve_documents]

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
user_input = "Can you summarize the main points about [Topic] from my document?"
events = app.stream({"messages": [HumanMessage(content=user_input)]})
for event in events:
    print(event)
