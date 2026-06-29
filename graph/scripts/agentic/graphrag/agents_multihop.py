"""
Agentic GraphRAG with Multi-Hop Reasoning Support
Demonstrates complex decision making using multi-layered graph traversal
"""

from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from tools_multihop import (
    search_unstructured_text, 
    query_graph_relationships, 
    query_multihop_relationships
)

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")

# Pack all three capabilities into the toolset
tools = [
    search_unstructured_text,          # For semantic text search
    query_graph_relationships,         # For simple, direct relationships
    query_multihop_relationships       # For complex, multi-hop reasoning
]

# Enhanced system instructions with multi-hop reasoning guidance
system_instructions = """You are a Neo4j Knowledge Assistant powered by Llama 3.2, specialized in the Iris botanical dataset with advanced multi-hop reasoning capabilities.

You have access to THREE tools:

1. `search_unstructured_text`: Use for conceptual questions about flower biology, morphology descriptions, botanical terminology, or general Iris information from text corpus.

2. `query_graph_relationships`: Use for SIMPLE, DIRECT graph queries:
   - Counting nodes (species, specimens)
   - Listing entities
   - Direct property lookups
   - Basic statistics

3. `query_multihop_relationships`: Use for COMPLEX reasoning requiring traversal across multiple entity levels:
   
   **When to use multi-hop:**
   - Detailed specimen characterization with full taxonomic context
   - Cross-species characteristic comparisons
   - Pattern discovery across measurement categories
   - Hierarchical aggregations (specimen → genus level)
   - Finding similar specimens across different species
   - Questions involving relationships like: Specimen → MeasurementGroup → CharacteristicType → Species → Genus
   
   **Multi-hop examples:**
   - "What are ALL the characteristics and taxonomy of Specimen_0?" (5-hop traversal)
   - "Compare sepal characteristics ACROSS all species" (3-hop aggregation)
   - "Find specimens with similar petal patterns but DIFFERENT species" (4-hop bidirectional)
   - "Which setosa specimens have large petal characteristics?" (filtered multi-hop)
   - "Show the distribution from genus to measurement level" (hierarchical rollup)

**Decision Making Strategy:**

For complex questions, PREFER multi-hop reasoning because it:
- Provides richer context by traversing relationships
- Enables comparison across taxonomic levels
- Reveals patterns not visible in single-hop queries
- Supports hierarchical analysis (bottom-up or top-down)

**Example Reasoning Chain:**

User: "Tell me about Specimen_0"

Simple approach (query_graph_relationships): 
  → Returns just measurements

Multi-hop approach (query_multihop_relationships):
  → Traverses Specimen_0 → Measurements → Characteristic Types → Species → Genus
  → Returns: measurements + size categories + characteristic groups + species classification + taxonomic hierarchy
  → BETTER for comprehensive understanding!

Always choose the tool that provides the MOST CONTEXT for the user's question. When in doubt, prefer multi-hop for specimen-specific or comparative questions.

DO NOT guess or make up information. Always rely on the tools to answer questions.
"""

# Compile the LangGraph engine with multi-hop support
agent_executor = create_react_agent(
    llm, 
    tools, 
    state_modifier=system_instructions
)


def run_interactive_session():
    """
    Interactive chat session demonstrating multi-hop reasoning capabilities
    """
    print("="*70)
    print("Iris Dataset GraphRAG Agent with Multi-Hop Reasoning")
    print("="*70)
    print("\nCapabilities:")
    print("  • Simple queries: species counts, specimen lists")
    print("  • Multi-hop reasoning: full specimen analysis, cross-species comparison")
    print("  • Pattern discovery: similar measurements across different species")
    print("  • Hierarchical analysis: genus → species → characteristics")
    print("\nExample questions:")
    print("  - 'What are all characteristics of Specimen_0?'")
    print("  - 'Compare sepal characteristics across all species'")
    print("  - 'Find specimens with similar petal patterns but different species'")
    print("  - 'Which setosa specimens have large petal characteristics?'")
    print("  - 'Show me the distribution of measurements'")
    print("\nType 'exit' to quit\n")
    print("="*70)
    
    while True:
        user_input = input("\n🌺 You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Iris GraphRAG Agent!")
            break
        
        if not user_input:
            continue
        
        try:
            # Invoke the agent with the user's question
            result = agent_executor.invoke({"messages": [("user", user_input)]})
            
            # Extract the final response
            if result and "messages" in result:
                final_message = result["messages"][-1]
                response = final_message.content if hasattr(final_message, 'content') else str(final_message)
                print(f"\n🤖 Assistant: {response}")
            else:
                print("\n🤖 Assistant: I couldn't process that request. Please try again.")
                
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            print("Please check that Neo4j is running and the graph is populated.")


if __name__ == "__main__":
    run_interactive_session()
