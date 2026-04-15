"""
Simple AI Agent Example

This example demonstrates a basic autonomous agent that can:
- Plan tasks
- Use tools
- Maintain memory
- Execute actions
"""

from typing import List, Dict, Any
import json


class SimpleTool:
    """Base class for agent tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        raise NotImplementedError


class CalculatorTool(SimpleTool):
    """Tool for performing calculations"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
    
    def execute(self, operation: str, a: float, b: float) -> str:
        """Execute a calculation"""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
        }
        
        if operation in operations:
            result = operations[operation](a, b)
            return f"Result: {result}"
        return f"Error: Unknown operation '{operation}'"


class Memory:
    """Simple memory system for the agent"""
    
    def __init__(self):
        self.short_term: List[Dict[str, Any]] = []
        self.long_term: List[Dict[str, Any]] = []
    
    def add_to_short_term(self, item: Dict[str, Any]):
        """Add item to short-term memory"""
        self.short_term.append(item)
        # Keep only last 10 items
        if len(self.short_term) > 10:
            self.short_term.pop(0)
    
    def add_to_long_term(self, item: Dict[str, Any]):
        """Add item to long-term memory"""
        self.long_term.append(item)
    
    def get_context(self) -> str:
        """Get current context from memory"""
        context = "Recent context:\n"
        for item in self.short_term[-5:]:
            context += f"- {item.get('type', 'unknown')}: {item.get('content', '')}\n"
        return context


class SimpleAgent:
    """A simple autonomous agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, SimpleTool] = {}
        self.memory = Memory()
        self.plan: List[str] = []
    
    def add_tool(self, tool: SimpleTool):
        """Add a tool to the agent's toolkit"""
        self.tools[tool.name] = tool
        print(f"[{self.name}] Tool added: {tool.name}")
    
    def create_plan(self, goal: str) -> List[str]:
        """Create a plan to achieve a goal"""
        print(f"\n[{self.name}] Creating plan for goal: {goal}")
        
        # This is a simplified planning process
        # In a real agent, this would use LLM reasoning
        if "calculate" in goal.lower():
            self.plan = [
                "Identify the calculation needed",
                "Use calculator tool",
                "Return result"
            ]
        else:
            self.plan = ["Analyze goal", "Execute action", "Verify result"]
        
        print(f"[{self.name}] Plan created:")
        for i, step in enumerate(self.plan, 1):
            print(f"  {i}. {step}")
        
        return self.plan
    
    def execute_step(self, step: str) -> str:
        """Execute a single step of the plan"""
        print(f"\n[{self.name}] Executing: {step}")
        
        # Record step in memory
        self.memory.add_to_short_term({
            "type": "step",
            "content": step,
            "status": "in_progress"
        })
        
        # Simulate step execution
        result = f"Completed: {step}"
        
        # Update memory
        self.memory.add_to_short_term({
            "type": "result",
            "content": result,
            "status": "completed"
        })
        
        return result
    
    def use_tool(self, tool_name: str, **kwargs) -> str:
        """Use a specific tool"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        print(f"[{self.name}] Using tool: {tool_name}")
        result = self.tools[tool_name].execute(**kwargs)
        
        # Record tool use in memory
        self.memory.add_to_short_term({
            "type": "tool_use",
            "tool": tool_name,
            "params": kwargs,
            "result": result
        })
        
        return result
    
    def run(self, goal: str) -> str:
        """Main execution loop for the agent"""
        print(f"\n{'='*60}")
        print(f"Agent '{self.name}' starting execution")
        print(f"Goal: {goal}")
        print(f"{'='*60}")
        
        # Create plan
        self.create_plan(goal)
        
        # Execute plan
        results = []
        for step in self.plan:
            result = self.execute_step(step)
            results.append(result)
        
        # Final result
        final_result = f"Goal achieved: {goal}"
        self.memory.add_to_long_term({
            "goal": goal,
            "plan": self.plan,
            "results": results,
            "status": "completed"
        })
        
        print(f"\n[{self.name}] Final result: {final_result}")
        print(f"{'='*60}\n")
        
        return final_result


def main():
    """Main function to demonstrate the agent"""
    
    # Create agent
    agent = SimpleAgent("MathBot")
    
    # Add tools
    calculator = CalculatorTool()
    agent.add_tool(calculator)
    
    # Execute a goal
    goal = "Calculate the sum of 15 and 27"
    agent.run(goal)
    
    # Use a tool directly
    print("\nDirect tool usage:")
    result = agent.use_tool("calculator", operation="add", a=15, b=27)
    print(f"Tool result: {result}")
    
    # Show memory context
    print("\n" + agent.memory.get_context())


if __name__ == "__main__":
    main()
