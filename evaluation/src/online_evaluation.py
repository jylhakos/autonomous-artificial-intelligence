"""
Online Evaluation and Monitoring

This module demonstrates how to implement online evaluation
for agents in production, monitoring real-time performance
and catching issues as they occur.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langsmith import Client
from langsmith.evaluation import evaluate
from typing import Annotated, Dict, Any
import os
import time
from datetime import datetime


# Initialize LangSmith client
client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))


@tool
def get_account_balance(account_id: Annotated[str, "The account ID"]) -> str:
    """Get the current account balance."""
    # Mock account data
    accounts = {
        "ACC001": "$5,250.00",
        "ACC002": "$12,750.00",
        "ACC003": "$850.00",
    }
    return accounts.get(account_id, "Account not found")


@tool
def get_recent_transactions(
    account_id: Annotated[str, "The account ID"],
    limit: Annotated[int, "Number of transactions to retrieve"] = 5,
) -> str:
    """Get recent transactions for an account."""
    # Mock transaction data
    transactions = {
        "ACC001": [
            "2024-04-14: -$50.00 (Coffee Shop)",
            "2024-04-13: +$2,000.00 (Salary Deposit)",
            "2024-04-12: -$125.00 (Grocery Store)",
        ],
        "ACC002": [
            "2024-04-14: -$500.00 (Rent Payment)",
            "2024-04-10: +$5,000.00 (Freelance Payment)",
        ],
    }
    
    account_txns = transactions.get(account_id, [])
    limited_txns = account_txns[:limit]
    
    if limited_txns:
        return "\n".join(limited_txns)
    else:
        return "No recent transactions found"


@tool
def transfer_funds(
    from_account: Annotated[str, "Source account ID"],
    to_account: Annotated[str, "Destination account ID"],
    amount: Annotated[float, "Amount to transfer"],
) -> str:
    """Transfer funds between accounts."""
    # Mock transfer - in production, this would be a real transaction
    return (
        f"Transfer successful!\n"
        f"From: {from_account}\n"
        f"To: {to_account}\n"
        f"Amount: ${amount:.2f}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def create_production_agent():
    """Create an agent for production deployment."""
    llm = ChatAnthropic(
        model="claude-sonnet-4",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    
    tools = [get_account_balance, get_recent_transactions, transfer_funds]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a banking assistant helping customers with their accounts. "
         "You can check balances, view transactions, and process transfers. "
         "Always confirm transfer details before executing. "
         "Be professional and security-conscious."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    
    return agent_executor


class ProductionMonitor:
    """
    Monitor agent performance in production.
    
    This class tracks:
    - Latency metrics
    - Error rates
    - Tool usage patterns
    - Response quality indicators
    """
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0.0,
            "tool_calls": {},
        }
    
    def track_request(
        self,
        query: str,
        response: Dict[str, Any],
        latency: float,
        success: bool,
    ):
        """Track a single agent request."""
        self.metrics["total_requests"] += 1
        self.metrics["total_latency"] += latency
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Track tool usage
        if "intermediate_steps" in response:
            for action, _ in response["intermediate_steps"]:
                tool_name = action.tool
                self.metrics["tool_calls"][tool_name] = (
                    self.metrics["tool_calls"].get(tool_name, 0) + 1
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        total = self.metrics["total_requests"]
        
        if total == 0:
            return {"message": "No requests tracked yet"}
        
        avg_latency = self.metrics["total_latency"] / total
        success_rate = (self.metrics["successful_requests"] / total) * 100
        
        return {
            "total_requests": total,
            "success_rate": f"{success_rate:.2f}%",
            "average_latency": f"{avg_latency:.2f}s",
            "error_rate": f"{(100 - success_rate):.2f}%",
            "tool_usage": self.metrics["tool_calls"],
        }
    
    def check_alerts(self) -> list:
        """Check for alert conditions."""
        alerts = []
        total = self.metrics["total_requests"]
        
        if total == 0:
            return alerts
        
        # Check error rate
        error_rate = (self.metrics["failed_requests"] / total) * 100
        if error_rate > 10:  # Alert if error rate > 10%
            alerts.append(f"HIGH ERROR RATE: {error_rate:.2f}%")
        
        # Check latency
        avg_latency = self.metrics["total_latency"] / total
        if avg_latency > 5:  # Alert if average latency > 5 seconds
            alerts.append(f"HIGH LATENCY: {avg_latency:.2f}s")
        
        return alerts


def simulate_production_traffic():
    """
    Simulate production traffic and monitor performance.
    
    This demonstrates online evaluation - monitoring agent
    behavior in real-time as it processes requests.
    """
    print("=" * 60)
    print("Online Evaluation and Production Monitoring")
    print("=" * 60)
    
    # Create agent and monitor
    agent = create_production_agent()
    monitor = ProductionMonitor()
    
    # Simulate production queries
    production_queries = [
        "What's the balance for account ACC001?",
        "Show me recent transactions for ACC002",
        "Transfer $100 from ACC001 to ACC002",
        "Check balance for ACC003 and show last 3 transactions",
        "What's my balance?",  # Intentionally vague to test error handling
    ]
    
    print("\nProcessing production traffic...")
    print("-" * 60)
    
    for i, query in enumerate(production_queries, 1):
        print(f"\n[Request {i}]: {query}")
        
        start_time = time.time()
        success = True
        
        try:
            # Run the agent
            response = agent.invoke({"input": query})
            print(f"[Response]: {response['output'][:100]}...")
            
        except Exception as e:
            print(f"[Error]: {str(e)}")
            response = {"output": str(e), "intermediate_steps": []}
            success = False
        
        latency = time.time() - start_time
        print(f"[Latency]: {latency:.2f}s")
        
        # Track in monitor
        monitor.track_request(query, response, latency, success)
        
        print("-" * 60)
    
    # Display monitoring summary
    print("\n" + "=" * 60)
    print("Production Monitoring Summary")
    print("=" * 60)
    
    summary = monitor.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check for alerts
    alerts = monitor.check_alerts()
    if alerts:
        print("\n" + "=" * 60)
        print("ALERTS")
        print("=" * 60)
        for alert in alerts:
            print(f"⚠️  {alert}")
    
    print("\n" + "=" * 60)
    print("All traces available in LangSmith for detailed analysis")
    print("View at: https://smith.langchain.com/")
    print("=" * 60)


def setup_online_evaluators():
    """
    Set up automated online evaluators in LangSmith.
    
    Online evaluators run automatically on production traces
    to catch issues in real-time.
    """
    print("\n" + "=" * 60)
    print("Setting Up Online Evaluators")
    print("=" * 60)
    
    print("""
Online evaluators can be configured in LangSmith to:

1. Latency Monitoring:
   - Alert when requests exceed threshold (e.g., > 5 seconds)
   - Track p50, p95, p99 latencies

2. Error Detection:
   - Catch tool call failures
   - Identify parsing errors
   - Monitor exception rates

3. Quality Checks:
   - Use LLM-as-judge for response quality
   - Verify tool usage patterns
   - Check for hallucinations

4. Business Metrics:
   - Track task completion rates
   - Monitor user satisfaction indicators
   - Measure conversation success

To set up online evaluators:
1. Go to LangSmith dashboard
2. Navigate to your project
3. Click "Evaluators" tab
4. Create rules for automated evaluation
5. Set up alerts and notifications

Production traces are automatically evaluated in real-time!
    """)


def main():
    """Main function for online evaluation demo."""
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("ERROR: LANGCHAIN_API_KEY not set")
        return
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    
    # Simulate production traffic
    simulate_production_traffic()
    
    # Show online evaluator setup
    setup_online_evaluators()


if __name__ == "__main__":
    main()
