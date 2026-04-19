from google.adk.agents import Agent
import datetime
from typing import Dict, Any

def get_order_status(order_id: str) -> Dict[str, Any]:
    """Retrieves order status information from the database.
    
    Args:
        order_id: The unique identifier for the order
        
    Returns:
        Dictionary containing order status and delivery information
    """
    # Simulate database lookup
    orders_db = {
        "12345": {
            "status": "shipped",
            "delivery_date": "2026-04-25",
            "tracking_number": "1Z999AA10123456784",
            "items": [
                {"name": "Laptop", "quantity": 1, "price": 1299.99}
            ]
        },
        "67890": {
            "status": "processing",
            "delivery_date": "2026-04-28",
            "tracking_number": None,
            "items": [
                {"name": "Wireless Mouse", "quantity": 2, "price": 29.99}
            ]
        }
    }
    
    order = orders_db.get(order_id)
    if order:
        return {
            "order_id": order_id,
            "status": order["status"],
            "delivery_date": order["delivery_date"],
            "tracking_number": order["tracking_number"],
            "items": order["items"]
        }
    else:
        return {
            "error": f"Order {order_id} not found",
            "suggestion": "Please verify the order ID and try again"
        }

def search_knowledge_base(query: str) -> str:
    """Searches the company knowledge base for product information.
    
    Args:
        query: The search query
        
    Returns:
        Relevant information from the knowledge base
    """
    # Simulate knowledge base search
    kb_articles = {
        "return policy": "We offer a 30-day return policy for most items. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.",
        "shipping": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) is available for an additional fee. Free shipping on orders over $50.",
        "warranty": "All electronics come with a 1-year manufacturer warranty. Extended warranties are available at checkout.",
        "payment": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. All transactions are encrypted and secure."
    }
    
    query_lower = query.lower()
    for topic, content in kb_articles.items():
        if topic in query_lower:
            return f"Topic: {topic.title()}\n\n{content}"
    
    return f"No specific article found for '{query}'. Please contact support for personalized assistance."

def escalate_to_human(reason: str, customer_id: str) -> Dict[str, Any]:
    """Creates a support ticket for human review.
    
    Args:
        reason: Explanation of why escalation is needed
        customer_id: The customer's unique identifier
        
    Returns:
        Ticket information
    """
    ticket_id = f"TICKET-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "ticket_id": ticket_id,
        "status": "pending_review",
        "assigned_to": "support_team",
        "priority": "normal",
        "reason": reason,
        "customer_id": customer_id,
        "created_at": datetime.datetime.now().isoformat(),
        "message": f"Ticket {ticket_id} has been created. A support agent will contact you within 24 hours."
    }

# Initialize the agent
support_agent = Agent(
    name="support_agent",
    model="gemini-2.0-flash",
    instruction="""You are a helpful customer support assistant for an e-commerce platform.
    
    Your capabilities:
    - Check order statuses by order ID
    - Search the knowledge base for product information
    - Escalate complex issues to human support agents
    
    Guidelines:
    - Always be polite and professional
    - Verify order IDs before checking status
    - Search knowledge base before escalating
    - Provide clear, concise responses
    - When escalating, explain the reason clearly
    - Set appropriate expectations for resolution times
    
    If you cannot help with a request, politely explain the limitation and offer alternatives.""",
    tools=[get_order_status, search_knowledge_base, escalate_to_human]
)

def run_agent(user_message: str, session_id: str = "default") -> str:
    """Run the support agent with a user message.
    
    Args:
        user_message: The user's input
        session_id: Session identifier for conversation continuity
        
    Returns:
        Agent's response
    """
    result = support_agent.run(user_message)
    return result.message

if __name__ == "__main__":
    # Test the agent
    test_queries = [
        "What is the status of order 12345?",
        "What is your return policy?",
        "I need help with a damaged item"
    ]
    
    print("Customer Support Agent - Test Mode\n")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = run_agent(query)
        print(f"Agent: {response}")
        print("-" * 50)
