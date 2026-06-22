import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Page Configuration ---
st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chat with RAG")
st.caption("Powered by Ollama (Llama 3.2), ChromaDB, and LangChain")

DB_DIR = "./chroma_db"
DOC_PATH = "document.txt"

# --- Functions cached by Streamlit to prevent reloading ---
@st.cache_resource
def initialize_rag_system():
    """Loads document, chunks it, embeds it, and returns the retriever."""
    if not os.path.exists(DOC_PATH):
        # Create a dummy file if it doesn't exist yet
        with open(DOC_PATH, "w") as f:
            f.write("The secret password for the underground facility is 'OLLAMA_RAG_2026'.\n")
            f.write("The facility is located three floors below the main library building.\n")
            f.write("Authorized personnel must wear blue badges at all times.\n")

    # Load and Chunk
    loader = TextLoader(DOC_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Initialize Vector DB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vector_store.as_retriever(search_kwargs={"k": 2})

@st.cache_resource
def get_rag_chain():
    """Initializes and returns the LLM chain pipeline."""
    llm = ChatOllama(model="llama3.2", temperature=0)
    template = """
    You are a helpful assistant. Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Note: retriever is passed dynamically during runtime invocation
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Initialize Resources ---
try:
    retriever = initialize_rag_system()
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Failed to connect to Ollama. Make sure the Ollama app is running! Error: {e}")
    st.stop()

# --- Streamlit Chat UI Logic ---
# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask a question about your local documents..."):
    
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Display assistant response with a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching local database and thinking..."):
            try:
                # Retrieve matching documents context
                docs = retriever.invoke(user_query)
                context_str = "\n\n".join(doc.page_content for doc in docs)
                
                # Generate final answer from LLM
                response = rag_chain.invoke({"context": context_str, "question": user_query})
                
                # Render response
                st.markdown(response)
                
                # Optional: Show the sources used inside an expander widget
                with st.expander("📚 View Retreived Context Sources"):
                    for i, doc in enumerate(docs):
                        st.info(f"**Source Chunk {i+1}:** {doc.page_content}")
                        
            except Exception as e:
                response = f"An error occurred: {e}"
                st.error(response)
                
    st.session_state.messages.append({"role": "assistant", "content": response})
