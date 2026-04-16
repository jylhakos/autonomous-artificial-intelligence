"""
Agent Observability Visualization Utilities

This module provides utilities for visualizing and tracking AI agent operations
throughout their lifecycle, including activity logging, response formatting, and
conversation trace visualization.
"""

import os
import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

try:
    from IPython.display import Markdown, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

try:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False


class AgentVisualizer:
    """
    Utility class for visualizing and tracking agent operations.
    
    This class provides methods for:
    - Logging agent activity with timestamps and severity levels
    - Displaying formatted agent responses
    - Visualizing complete conversation traces
    - Managing activity context across agent sessions
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the agent visualizer.
        
        Args:
            session_id: Optional session identifier for tracking related operations
        """
        load_dotenv()
        self.activity_log: List[Dict[str, Any]] = []
        self.session_id = session_id or self._generate_session_id()
        self.start_time = datetime.datetime.now()
        
    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        return f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def print_activity(self, message: str, level: str = "INFO", context: Optional[Dict] = None):
        """
        Log agent activity with timestamp and severity level.
        
        Args:
            message: Activity message to log
            level: Severity level (INFO, DEBUG, WARNING, ERROR)
            context: Optional context dictionary with additional metadata
        """
        timestamp = datetime.datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "session_id": self.session_id,
            "context": context or {}
        }
        
        self.activity_log.append(entry)
        
        # Format output with color coding based on level
        level_colors = {
            "DEBUG": "\033[36m",    # Cyan
            "INFO": "\033[32m",     # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "RESET": "\033[0m"      # Reset
        }
        
        color = level_colors.get(level, level_colors["RESET"])
        reset = level_colors["RESET"]
        
        formatted_entry = f"{color}[{timestamp}] [{level}]{reset} {message}"
        print(formatted_entry)
        
        if context:
            print(f"  Context: {context}")
    
    def display_agent_response(self, response: Any, format_type: str = "auto"):
        """
        Display formatted agent response.
        
        Args:
            response: Agent response to display (can be various types)
            format_type: Display format ('auto', 'markdown', 'text', 'json')
        """
        try:
            if format_type == "markdown" and IPYTHON_AVAILABLE:
                if hasattr(response, 'content'):
                    display(Markdown(response.content))
                elif isinstance(response, str):
                    display(Markdown(response))
                else:
                    display(response)
            elif format_type == "json":
                import json
                if isinstance(response, (dict, list)):
                    print(json.dumps(response, indent=2))
                else:
                    print(response)
            else:
                # Auto-detect or plain text
                if hasattr(response, 'content'):
                    print(response.content)
                else:
                    print(response)
                    
        except Exception as e:
            self.print_activity(
                f"Failed to display response: {str(e)}",
                level="ERROR",
                context={"response_type": type(response).__name__}
            )
    
    def visualize_conversation(self, conversation_id: Optional[str] = None):
        """
        Visualize complete conversation trace with all activities.
        
        Args:
            conversation_id: Optional conversation identifier for display
        """
        conv_id = conversation_id or self.session_id
        
        print("\n" + "=" * 80)
        print(f"CONVERSATION TRACE: {conv_id}")
        print(f"Session Duration: {self._get_session_duration()}")
        print(f"Total Activities: {len(self.activity_log)}")
        print("=" * 80 + "\n")
        
        for entry in self.activity_log:
            timestamp = entry["timestamp"]
            level = entry["level"]
            message = entry["message"]
            context = entry.get("context", {})
            
            print(f"[{timestamp}] [{level}]")
            print(f"  > {message}")
            
            if context:
                print(f"  Context:")
                for key, value in context.items():
                    print(f"    - {key}: {value}")
            print()
        
        print("=" * 80)
        print("END TRACE")
        print("=" * 80 + "\n")
    
    def _get_session_duration(self) -> str:
        """Calculate and format session duration."""
        duration = datetime.datetime.now() - self.start_time
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def reset_activity_context(self):
        """Reset activity log for new conversation while preserving session."""
        self.print_activity("Resetting activity context", level="INFO")
        self.activity_log = []
        self.start_time = datetime.datetime.now()
    
    def export_trace(self, filepath: str, format: str = "json"):
        """
        Export conversation trace to file.
        
        Args:
            filepath: Path to export file
            format: Export format ('json', 'csv', 'txt')
        """
        import json
        
        try:
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump({
                        "session_id": self.session_id,
                        "start_time": self.start_time.isoformat(),
                        "duration": self._get_session_duration(),
                        "activities": self.activity_log
                    }, f, indent=2)
            elif format == "csv":
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Level", "Message", "Context"])
                    for entry in self.activity_log:
                        writer.writerow([
                            entry["timestamp"],
                            entry["level"],
                            entry["message"],
                            str(entry.get("context", ""))
                        ])
            elif format == "txt":
                with open(filepath, 'w') as f:
                    f.write(f"Session: {self.session_id}\n")
                    f.write(f"Duration: {self._get_session_duration()}\n\n")
                    for entry in self.activity_log:
                        f.write(f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}\n")
                        if entry.get("context"):
                            f.write(f"  Context: {entry['context']}\n")
                        f.write("\n")
            
            self.print_activity(
                f"Trace exported to {filepath}",
                level="INFO",
                context={"format": format}
            )
            
        except Exception as e:
            self.print_activity(
                f"Failed to export trace: {str(e)}",
                level="ERROR",
                context={"filepath": filepath, "format": format}
            )
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current session.
        
        Returns:
            Dictionary containing session statistics
        """
        level_counts = {}
        for entry in self.activity_log:
            level = entry["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_activities": len(self.activity_log),
            "duration": self._get_session_duration(),
            "level_breakdown": level_counts,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.datetime.now().isoformat()
        }


# Convenience functions for global usage
_global_visualizer: Optional[AgentVisualizer] = None


def get_visualizer() -> AgentVisualizer:
    """Get or create global visualizer instance."""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = AgentVisualizer()
    return _global_visualizer


def print_activity(message: str, level: str = "INFO", context: Optional[Dict] = None):
    """Global convenience function for logging activity."""
    get_visualizer().print_activity(message, level, context)


def display_agent_response(response: Any, format_type: str = "auto"):
    """Global convenience function for displaying responses."""
    get_visualizer().display_agent_response(response, format_type)


def visualize_conversation(conversation_id: Optional[str] = None):
    """Global convenience function for visualizing conversations."""
    get_visualizer().visualize_conversation(conversation_id)


def reset_activity_context():
    """Global convenience function for resetting context."""
    get_visualizer().reset_activity_context()


# Example usage
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = AgentVisualizer(session_id="demo_session")
    
    # Log various activities
    visualizer.print_activity("Starting weather data retrieval", level="INFO")
    visualizer.print_activity("Connecting to API", level="DEBUG", context={"api": "openweathermap"})
    visualizer.print_activity("Retrieved weather data successfully", level="INFO", 
                            context={"city": "London", "temperature": 15.5})
    
    # Display response
    visualizer.display_agent_response("Weather data retrieved: 15.5°C in London")
    
    # Show conversation trace
    visualizer.visualize_conversation()
    
    # Get statistics
    stats = visualizer.get_summary_statistics()
    print("\nSession Statistics:")
    print(stats)
