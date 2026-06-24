from langchain_core.tools import tool
from langchain.chains import GraphCypherQAChain
from langchain_ollama import ChatOllama

# Initialize local LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# Tool A: Vector Retrieval for unstructured text matching
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def search_unstructured_text(query: str) -> str:
    """Useful for answering semantic questions, finding main takeaways, summaries, 
    or reading raw text blocks from document.txt."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Tool B: Graph Cypher execution for structured relationship patterns
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

@tool
def query_graph_relationships(query: str) -> str:
    """Useful for counting nodes, analyzing explicit links/connections, paths, 
    or running advanced Cypher pattern-matching queries against the database."""
    try:
        return cypher_chain.run(query)
    except Exception as e:
        return f"Error executing Cypher query: {e}"
