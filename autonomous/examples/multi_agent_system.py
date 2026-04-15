"""
Multi-Agent System Example

This example demonstrates a supervisor-based multi-agent system where:
- A supervisor agent coordinates multiple specialized agents
- Each agent has specific expertise
- Agents collaborate to solve complex tasks
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class AgentStatus(Enum):
    """Status of an agent"""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Task:
    """Represents a task to be executed"""
    id: str
    description: str
    agent_type: str
    status: AgentStatus = AgentStatus.IDLE
    result: Any = None


class SpecializedAgent:
    """Base class for specialized agents"""
    
    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.status = AgentStatus.IDLE
    
    def can_handle(self, task: Task) -> bool:
        """Check if this agent can handle the given task"""
        return task.agent_type == self.expertise
    
    def execute(self, task: Task) -> Any:
        """Execute a task"""
        print(f"[{self.name}] Executing task: {task.description}")
        self.status = AgentStatus.WORKING
        
        # Simulate task execution
        result = f"Task '{task.description}' completed by {self.name}"
        
        self.status = AgentStatus.COMPLETED
        return result


class ResearchAgent(SpecializedAgent):
    """Agent specialized in research tasks"""
    
    def __init__(self):
        super().__init__("ResearchBot", "research")
    
    def execute(self, task: Task) -> Any:
        """Execute research task"""
        print(f"[{self.name}] Conducting research on: {task.description}")
        self.status = AgentStatus.WORKING
        
        # Simulate research
        findings = {
            "query": task.description,
            "sources": ["Source A", "Source B", "Source C"],
            "summary": f"Research findings for: {task.description}",
            "confidence": 0.85
        }
        
        self.status = AgentStatus.COMPLETED
        return findings


class CodeAgent(SpecializedAgent):
    """Agent specialized in coding tasks"""
    
    def __init__(self):
        super().__init__("CodeBot", "coding")
    
    def execute(self, task: Task) -> Any:
        """Execute coding task"""
        print(f"[{self.name}] Writing code for: {task.description}")
        self.status = AgentStatus.WORKING
        
        # Simulate code generation
        code = f"""
def solution():
    # Generated code for: {task.description}
    result = "Implementation complete"
    return result
"""
        
        self.status = AgentStatus.COMPLETED
        return {
            "code": code,
            "language": "python",
            "tests_passed": True
        }


class AnalysisAgent(SpecializedAgent):
    """Agent specialized in data analysis"""
    
    def __init__(self):
        super().__init__("AnalysisBot", "analysis")
    
    def execute(self, task: Task) -> Any:
        """Execute analysis task"""
        print(f"[{self.name}] Analyzing: {task.description}")
        self.status = AgentStatus.WORKING
        
        # Simulate analysis
        analysis = {
            "task": task.description,
            "metrics": {
                "accuracy": 0.92,
                "performance": 0.88,
                "reliability": 0.95
            },
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2"
            ]
        }
        
        self.status = AgentStatus.COMPLETED
        return analysis


class SupervisorAgent:
    """Supervisor agent that coordinates specialized agents"""
    
    def __init__(self):
        self.name = "Supervisor"
        self.agents: List[SpecializedAgent] = []
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
    
    def register_agent(self, agent: SpecializedAgent):
        """Register a specialized agent"""
        self.agents.append(agent)
        print(f"[{self.name}] Registered agent: {agent.name} ({agent.expertise})")
    
    def decompose_goal(self, goal: str) -> List[Task]:
        """Break down a complex goal into tasks"""
        print(f"\n[{self.name}] Decomposing goal: {goal}")
        
        # Simple task decomposition logic
        tasks = []
        
        if "analyze" in goal.lower():
            tasks.append(Task("1", "Research background information", "research"))
            tasks.append(Task("2", "Perform detailed analysis", "analysis"))
            tasks.append(Task("3", "Generate analysis report", "coding"))
        else:
            tasks.append(Task("1", f"Primary task: {goal}", "research"))
        
        print(f"[{self.name}] Created {len(tasks)} tasks")
        return tasks
    
    def assign_task(self, task: Task) -> SpecializedAgent:
        """Assign task to appropriate agent"""
        for agent in self.agents:
            if agent.can_handle(task) and agent.status == AgentStatus.IDLE:
                print(f"[{self.name}] Assigning task '{task.id}' to {agent.name}")
                return agent
        
        print(f"[{self.name}] No available agent for task '{task.id}'")
        return None
    
    def execute_workflow(self, goal: str) -> Dict[str, Any]:
        """Execute a multi-agent workflow"""
        print(f"\n{'='*70}")
        print(f"Starting Multi-Agent Workflow")
        print(f"Goal: {goal}")
        print(f"{'='*70}\n")
        
        # Step 1: Decompose goal into tasks
        tasks = self.decompose_goal(goal)
        self.task_queue = tasks
        
        # Step 2: Execute tasks
        results = {}
        
        for task in tasks:
            # Assign task to agent
            agent = self.assign_task(task)
            
            if agent:
                # Execute task
                task.status = AgentStatus.WORKING
                result = agent.execute(task)
                task.result = result
                task.status = AgentStatus.COMPLETED
                
                results[task.id] = {
                    "task": task.description,
                    "agent": agent.name,
                    "result": result
                }
                
                self.completed_tasks.append(task)
            else:
                task.status = AgentStatus.ERROR
                results[task.id] = {
                    "task": task.description,
                    "error": "No agent available"
                }
        
        # Step 3: Aggregate results
        print(f"\n[{self.name}] Aggregating results...")
        final_result = {
            "goal": goal,
            "total_tasks": len(tasks),
            "completed": len(self.completed_tasks),
            "results": results,
            "status": "success" if len(self.completed_tasks) == len(tasks) else "partial"
        }
        
        print(f"[{self.name}] Workflow completed!")
        print(f"{'='*70}\n")
        
        return final_result


def main():
    """Main function to demonstrate multi-agent system"""
    
    # Create supervisor
    supervisor = SupervisorAgent()
    
    # Create and register specialized agents
    research_agent = ResearchAgent()
    code_agent = CodeAgent()
    analysis_agent = AnalysisAgent()
    
    supervisor.register_agent(research_agent)
    supervisor.register_agent(code_agent)
    supervisor.register_agent(analysis_agent)
    
    # Execute a complex goal
    goal = "Analyze the performance of our new recommendation system"
    result = supervisor.execute_workflow(goal)
    
    # Display results
    print("\nFinal Results:")
    print(f"Goal: {result['goal']}")
    print(f"Status: {result['status']}")
    print(f"Tasks completed: {result['completed']}/{result['total_tasks']}")
    
    print("\nTask Details:")
    for task_id, task_info in result['results'].items():
        print(f"\nTask {task_id}: {task_info['task']}")
        print(f"  Agent: {task_info.get('agent', 'N/A')}")
        print(f"  Result: {task_info.get('result', task_info.get('error'))}")


if __name__ == "__main__":
    main()
