#!/usr/bin/env python3
"""
Blog Writing Multi-Agent System using CrewAI
Demonstrates orchestration of AI agents for content creation.
"""

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os
import sys


class BlogAgentOrchestrator:
    """Orchestrates a multi-agent pipeline for blog writing."""
    
    def __init__(self, topic, model_provider="openai"):
        """
        Initialize the blog agent orchestrator.
        
        Args:
            topic: The blog topic to write about
            model_provider: LLM provider ('openai', 'ollama', or 'bedrock')
        """
        self.topic = topic
        self.model_provider = model_provider
        self.search_tool = None
        self.scrape_tool = None
        
        # Initialize tools if API keys are available
        if os.getenv("SERPER_API_KEY"):
            self.search_tool = SerperDevTool()
        
        self.scrape_tool = ScrapeWebsiteTool()
    
    def get_llm_config(self):
        """
        Get LLM configuration based on provider.
        
        Returns:
            LLM configuration dictionary or None for default
        """
        if self.model_provider == "ollama":
            return {
                "model": "ollama/mistral",
                "base_url": "http://localhost:11434"
            }
        elif self.model_provider == "bedrock":
            return {
                "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                "aws_region_name": "us-east-1"
            }
        # Default to OpenAI-compatible models
        return None
    
    def create_planner_agent(self):
        """Create the content planning agent."""
        tools = []
        if self.search_tool:
            tools.append(self.search_tool)
        if self.scrape_tool:
            tools.append(self.scrape_tool)
        
        llm_config = self.get_llm_config()
        
        return Agent(
            role="Content Planner",
            goal=f"Plan engaging and factually accurate content on {self.topic}",
            backstory=f"""You are an expert content strategist with years of experience 
            in planning blog articles. You excel at researching the topic '{self.topic}' 
            and identifying the latest trends, key players, and noteworthy developments.
            You understand audience needs and create detailed content outlines that 
            serve as blueprints for exceptional articles. Your research forms the 
            foundation for writers to craft compelling narratives.""",
            allow_delegation=False,
            tools=tools,
            llm=llm_config,
            verbose=True
        )
    
    def create_writer_agent(self):
        """Create the content writing agent."""
        llm_config = self.get_llm_config()
        
        return Agent(
            role="Content Writer",
            goal=f"Write an engaging, well-researched blog article about {self.topic}",
            backstory=f"""You are a skilled content writer specializing in creating 
            compelling blog articles. You have a talent for transforming research and 
            outlines into engaging narratives that captivate readers. You write in a 
            clear, accessible style while maintaining technical accuracy. Your articles 
            are well-structured with smooth transitions between ideas, making complex 
            topics about '{self.topic}' easy to understand.""",
            allow_delegation=False,
            llm=llm_config,
            verbose=True
        )
    
    def create_editor_agent(self):
        """Create the editing and quality assurance agent."""
        llm_config = self.get_llm_config()
        
        return Agent(
            role="Content Editor",
            goal=f"Edit and refine the blog article about {self.topic} to publication quality",
            backstory=f"""You are a meticulous editor with a keen eye for detail and 
            a commitment to quality. You review articles for grammar, clarity, flow, 
            and impact. You ensure the content aligns with best practices for blog 
            writing, including proper structure, engaging headlines, and effective 
            calls to action. You polish the article until it shines, making it ready 
            for publication.""",
            allow_delegation=False,
            llm=llm_config,
            verbose=True
        )
    
    def create_planning_task(self, agent):
        """Create the content planning task."""
        return Task(
            description=f"""Conduct comprehensive research and planning for a blog article about {self.topic}.
            
            Your responsibilities:
            1. Research the latest trends, developments, and key information about {self.topic}
            2. Identify the target audience and their interests, pain points, and questions
            3. Analyze what makes content engaging for this audience
            4. Develop a detailed content outline with:
               - An attention-grabbing headline
               - Introduction hook
               - Main sections with key points
               - Supporting facts and data
               - Conclusion with call to action
            5. Identify relevant SEO keywords to include
            6. List credible sources and references to cite
            
            Focus on creating a comprehensive blueprint that will guide the writer.""",
            expected_output=f"""A detailed content plan document containing:
            - Proposed headline and variations
            - Target audience profile
            - Comprehensive outline with section headers and key points
            - List of SEO keywords
            - Research findings and sources
            - Content tone and style guidelines""",
            agent=agent
        )
    
    def create_writing_task(self, agent, context_tasks):
        """Create the content writing task."""
        return Task(
            description=f"""Write a complete blog article about {self.topic} based on the content plan.
            
            Your responsibilities:
            1. Follow the outline and structure from the content plan
            2. Write an engaging introduction that hooks the reader
            3. Develop each main section with clear explanations and examples
            4. Incorporate the research findings and data naturally
            5. Use the identified SEO keywords appropriately
            6. Maintain a consistent, engaging tone throughout
            7. Write clear transitions between sections
            8. Create a compelling conclusion with a call to action
            9. Aim for 800-1200 words
            
            Write in a clear, accessible style that makes the topic easy to understand.""",
            expected_output=f"""A complete blog article draft about {self.topic} containing:
            - Engaging headline
            - Well-structured content with introduction, body sections, and conclusion
            - Proper use of SEO keywords
            - Citations and references where appropriate
            - Clear call to action
            - 800-1200 words""",
            agent=agent,
            context=context_tasks
        )
    
    def create_editing_task(self, agent, context_tasks):
        """Create the editing and refinement task."""
        return Task(
            description=f"""Edit and refine the blog article about {self.topic} to publication quality.
            
            Your responsibilities:
            1. Review the article for grammar, spelling, and punctuation errors
            2. Improve sentence structure and clarity
            3. Enhance flow and transitions between paragraphs
            4. Verify factual accuracy and proper citations
            5. Strengthen the headline for maximum impact
            6. Ensure SEO keywords are naturally integrated
            7. Optimize the introduction and conclusion
            8. Format the article properly with headers and sections
            9. Add any missing elements for completeness
            10. Provide a final polished version ready for publication
            
            Ensure the final article is engaging, accurate, and professional.""",
            expected_output=f"""A publication-ready blog article about {self.topic} with:
            - Polished, error-free content
            - Strong headline
            - Excellent flow and readability
            - Proper formatting and structure
            - Optimized for SEO
            - Professional quality suitable for immediate publication""",
            agent=agent,
            context=context_tasks
        )
    
    def run(self, output_file="blog_output.md"):
        """
        Execute the multi-agent blog writing pipeline.
        
        Args:
            output_file: Path to save the final blog article
            
        Returns:
            The final blog article content
        """
        print(f"\n{'='*60}")
        print(f"Starting Multi-Agent Blog Writing Pipeline")
        print(f"Topic: {self.topic}")
        print(f"LLM Provider: {self.model_provider}")
        print(f"{'='*60}\n")
        
        # Create agents
        planner_agent = self.create_planner_agent()
        writer_agent = self.create_writer_agent()
        editor_agent = self.create_editor_agent()
        
        # Create tasks
        planning_task = self.create_planning_task(planner_agent)
        writing_task = self.create_writing_task(writer_agent, [planning_task])
        editing_task = self.create_editing_task(editor_agent, [writing_task])
        
        # Create the crew
        crew = Crew(
            agents=[planner_agent, writer_agent, editor_agent],
            tasks=[planning_task, writing_task, editing_task],
            verbose=True,
            memory=True,
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                }
            },
            cache=True,
            process=Process.sequential
        )
        
        # Execute the pipeline
        print("\nExecuting multi-agent pipeline...\n")
        result = crew.kickoff()
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(result))
        
        print(f"\n{'='*60}")
        print(f"Blog article completed and saved to: {output_file}")
        print(f"{'='*60}\n")
        
        return result


def main():
    """Main execution function."""
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OLLAMA_HOST"):
        print("\nWARNING: No LLM provider configured!")
        print("\nTo run this example, you need to set up one of the following:")
        print("\n1. OpenAI (default):")
        print("   export OPENAI_API_KEY='your-api-key'")
        print("\n2. Ollama (local, free):")
        print("   Install Ollama from https://ollama.ai")
        print("   Run: ollama pull mistral")
        print("   The script will use it automatically")
        print("\n3. AWS Bedrock:")
        print("   Configure AWS credentials and set AWS_REGION")
        print("\nOptional - For web search capability:")
        print("   export SERPER_API_KEY='your-serper-api-key'")
        print("   Get free API key at https://serper.dev\n")
        sys.exit(1)
    
    # Default topic
    topic = "AI Agent Orchestration in Software Development"
    
    # Determine which provider to use
    if os.getenv("OPENAI_API_KEY"):
        provider = "openai"
    elif os.getenv("OLLAMA_HOST") or os.path.exists("/usr/local/bin/ollama"):
        provider = "ollama"
    else:
        provider = "openai"
    
    # Allow custom topic from command line
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    
    # Create and run orchestrator
    orchestrator = BlogAgentOrchestrator(topic, model_provider=provider)
    orchestrator.run()


if __name__ == "__main__":
    main()
