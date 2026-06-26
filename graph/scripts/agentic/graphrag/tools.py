from langchain_core.tools import tool
from langchain.chains import GraphCypherQAChain
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import OllamaEmbeddings

# Neo4j connection configuration
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# Initialize local LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# Initialize Neo4j graph connection
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD)

# Initialize embeddings for vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize vector store from Neo4j
vector_store = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="document_vector_index",
    node_label="DocumentChunk",
    text_node_property="text",
    embedding_node_property="embedding"
)

# Tool A: Vector Retrieval for unstructured text matching
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def search_unstructured_text(query: str) -> str:
    """Useful for answering semantic questions about Iris dataset, finding morphological characteristics, 
    species information, or reading raw text blocks from the Iris document."""
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
    """Useful for counting nodes, analyzing explicit links/connections between Iris species, 
    specimens, and measurements, paths, or running advanced Cypher pattern-matching 
    queries against the Neo4j database."""
    try:
        return cypher_chain.run(query)
    except Exception as e:
        return f"Error executing Cypher query: {e}"
