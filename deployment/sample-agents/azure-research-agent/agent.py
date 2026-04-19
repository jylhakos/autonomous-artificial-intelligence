from azure.identity import DefaultAzureCredential
from azure.ai.openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Any
import os
import json

class ResearchAgent:
    """Academic research assistant powered by Azure AI."""
    
    def __init__(self):
        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version="2024-02-01"
        )
        
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4-turbo')
        
        search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        search_key = os.getenv('AZURE_SEARCH_KEY')
        search_index = os.getenv('AZURE_SEARCH_INDEX', 'research-papers')
        
        if search_endpoint and search_key:
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=search_index,
                credential=AzureKeyCredential(search_key)
            )
        else:
            self.search_client = None
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search academic documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of document results with metadata
        """
        if not self.search_client:
            return [{
                "id": "demo-1",
                "title": "Introduction to Quantum Computing",
                "authors": ["Smith, J.", "Jones, A."],
                "year": 2023,
                "abstract": "This paper provides a comprehensive overview of quantum computing...",
                "citations": 45,
                "source": "Journal of Quantum Science"
            }]
        
        try:
            results = self.search_client.search(
                search_text=query,
                top=top_k,
                select=["id", "title", "authors", "year", "abstract", "citations", "source"]
            )
            
            documents = []
            for result in results:
                documents.append({
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "authors": result.get("authors", []),
                    "year": result.get("year"),
                    "abstract": result.get("abstract", ""),
                    "citations": result.get("citations", 0),
                    "source": result.get("source", ""),
                    "relevance_score": result.get("@search.score", 0)
                })
            
            return documents
        
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a research document.
        
        Args:
            document: Document to analyze
            
        Returns:
            Analysis results
        """
        analysis_prompt = f"""Analyze this research document:

Title: {document.get('title', 'N/A')}
Authors: {', '.join(document.get('authors', []))}
Year: {document.get('year', 'N/A')}
Abstract: {document.get('abstract', 'N/A')}

Provide:
1. Key findings (3-5 bullet points)
2. Methodology used
3. Significance and impact
4. Limitations
5. Related research areas

Format as JSON with keys: findings, methodology, significance, limitations, related_areas"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis = {"raw_analysis": analysis_text}
            
            return {
                "document_id": document.get("id"),
                "analysis": analysis
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_research_summary(self, topic: str, documents: List[Dict[str, Any]]) -> str:
        """Generate a research summary from multiple documents.
        
        Args:
            topic: Research topic
            documents: List of relevant documents
            
        Returns:
            Formatted research summary
        """
        docs_text = "\n\n".join([
            f"Document {i+1}:\n"
            f"Title: {doc.get('title', 'N/A')}\n"
            f"Authors: {', '.join(doc.get('authors', []))}\n"
            f"Year: {doc.get('year', 'N/A')}\n"
            f"Abstract: {doc.get('abstract', 'N/A')[:300]}..."
            for i, doc in enumerate(documents[:5])
        ])
        
        summary_prompt = f"""Generate a comprehensive research summary on: {topic}

Based on these documents:
{docs_text}

Structure your summary as:
1. Overview of the field
2. Key findings across studies
3. Common methodologies
4. Current trends and gaps
5. Future research directions

Include proper citations in [Author, Year] format."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer specializing in research synthesis."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.5,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def ask_question(self, question: str, context: str = "") -> str:
        """Answer a research question.
        
        Args:
            question: Research question
            context: Optional context from previous searches
            
        Returns:
            Answer with citations
        """
        system_message = """You are a helpful research assistant. When answering questions:
- Provide evidence-based responses
- Include citations when referencing specific studies
- Acknowledge limitations in your knowledge
- Suggest further reading when appropriate
- Use academic language but remain accessible"""

        messages = [{"role": "system", "content": system_message}]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"Context from research documents:\n{context}"
            })
        
        messages.append({"role": "user", "content": question})
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error processing question: {str(e)}"

research_agent = ResearchAgent()

if __name__ == "__main__":
    print("Research Assistant Agent - Test Mode\n")
    print("=" * 50)
    
    # Test document search
    print("\n1. Document Search:")
    results = research_agent.search_documents("quantum computing", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n  {i}. {doc.get('title', 'N/A')}")
        print(f"     Authors: {', '.join(doc.get('authors', []))}")
        print(f"     Year: {doc.get('year', 'N/A')}")
    
    # Test document analysis
    print("\n2. Document Analysis:")
    if results:
        analysis = research_agent.analyze_document(results[0])
        print(json.dumps(analysis, indent=2))
    
    # Test research summary
    print("\n3. Research Summary:")
    summary = research_agent.generate_research_summary(
        "quantum computing applications",
        results
    )
    print(summary[:500] + "...")
    
    # Test question answering
    print("\n4. Question Answering:")
    question = "What are the main challenges in quantum computing?"
    answer = research_agent.ask_question(question)
    print(f"Q: {question}")
    print(f"A: {answer[:300]}...")
