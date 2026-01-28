"""
State definitions for the LangGraph reasoning agent.
"""
from typing import TypedDict, List, Optional, Annotated
from langchain.schema import BaseMessage
from operator import add


class AgentState(TypedDict):
    """
    State structure for the LangGraph agent.
    
    This defines the data that flows through the graph nodes.
    Each node can read from and write to this state.
    """
    
    # User query
    query: str
    
    # Messages history for conversation
    messages: Annotated[List[BaseMessage], add]
    
    # Retrieved documents from RAG
    retrieved_docs: Optional[List[str]]
    
    # Web search results
    web_results: Optional[List[str]]
    
    # Decision on which tool to use
    next_action: Optional[str]
    
    # Final answer
    answer: Optional[str]
    
    # Iteration counter to prevent infinite loops
    iteration: int
    
    # Maximum iterations allowed
    max_iterations: int


def create_initial_state(query: str, max_iterations: int = 5) -> AgentState:
    """
    Create an initial state for the agent.
    
    Args:
        query: User's query
        max_iterations: Maximum number of iterations
        
    Returns:
        AgentState initialized with the query
    """
    return AgentState(
        query=query,
        messages=[],
        retrieved_docs=None,
        web_results=None,
        next_action=None,
        answer=None,
        iteration=0,
        max_iterations=max_iterations
    )
