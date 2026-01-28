"""
Graph nodes for the LangGraph reasoning agent.
"""
import logging
from typing import Dict, Any
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from src.reasoning.state import AgentState
from src.reasoning.tools import TavilySearchTool, LocalRetrievalTool

logger = logging.getLogger(__name__)

# Constants for content truncation
MAX_CONTEXT_LENGTH = 500


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Router node that decides which action to take next.
    
    Analyzes the query and decides whether to use RAG retrieval,
    web search, or go directly to answer generation.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with next_action set
    """
    query = state["query"]
    iteration = state.get("iteration", 0)
    
    # Simple heuristic-based routing
    # In production, this could use an LLM to make the decision
    
    # Check for keywords that suggest web search
    # Use current year dynamically to avoid hardcoding
    current_year = datetime.now().year
    future_years = [str(current_year + i) for i in range(5)]
    web_keywords = ["current", "latest", "news", "today", "recent"] + future_years
    needs_web = any(keyword in query.lower() for keyword in web_keywords)
    
    # If we don't have any results yet, try local retrieval first
    if state.get("retrieved_docs") is None and not needs_web:
        next_action = "retrieve"
    # If we need web search and haven't done it yet
    elif needs_web and state.get("web_results") is None:
        next_action = "search_web"
    # Otherwise, generate answer
    else:
        next_action = "generate"
    
    logger.debug(f"Router decided: {next_action} (iteration {iteration})")
    
    return {
        **state,
        "next_action": next_action,
        "iteration": iteration + 1
    }


def rag_retrieval_node(state: AgentState, retrieval_tool: LocalRetrievalTool) -> Dict[str, Any]:
    """
    RAG retrieval node that fetches relevant documents from local database.
    
    Args:
        state: Current agent state
        retrieval_tool: LocalRetrievalTool instance
        
    Returns:
        Updated state with retrieved_docs populated
    """
    query = state["query"]
    
    # Retrieve relevant documents
    documents = retrieval_tool.retrieve(query)
    
    # Add message to conversation history
    message = HumanMessage(content=f"Retrieved {len(documents)} documents from local database")
    
    logger.info(f"RAG retrieval: {len(documents)} documents")
    
    return {
        **state,
        "retrieved_docs": documents,
        "messages": state.get("messages", []) + [message]
    }


def web_search_node(state: AgentState, search_tool: TavilySearchTool) -> Dict[str, Any]:
    """
    Web search node that fetches information from the internet.
    
    Args:
        state: Current agent state
        search_tool: TavilySearchTool instance
        
    Returns:
        Updated state with web_results populated
    """
    query = state["query"]
    
    # Perform web search
    results = search_tool.search(query)
    
    # Add message to conversation history
    message = HumanMessage(content=f"Retrieved {len(results)} results from web search")
    
    logger.info(f"Web search: {len(results)} results")
    
    return {
        **state,
        "web_results": results,
        "messages": state.get("messages", []) + [message]
    }


def generate_answer_node(state: AgentState, llm) -> Dict[str, Any]:
    """
    Answer generation node that synthesizes information and produces final answer.
    
    Args:
        state: Current agent state
        llm: Language model instance
        
    Returns:
        Updated state with answer populated
    """
    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])
    web_results = state.get("web_results", [])
    
    # Prepare context from retrieved information
    context_parts = []
    
    if retrieved_docs:
        context_parts.append("## Local Knowledge Base:")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            # Truncate with note about truncation
            truncated = doc[:MAX_CONTEXT_LENGTH]
            if len(doc) > MAX_CONTEXT_LENGTH:
                truncated += "... [truncated]"
            context_parts.append(f"{i}. {truncated}")
    
    if web_results:
        context_parts.append("\n## Web Search Results:")
        for i, result in enumerate(web_results[:3], 1):
            # Truncate with note about truncation
            truncated = result[:MAX_CONTEXT_LENGTH]
            if len(result) > MAX_CONTEXT_LENGTH:
                truncated += "... [truncated]"
            context_parts.append(f"{i}. {truncated}")
    
    context = "\n".join(context_parts) if context_parts else "No additional context available."
    
    # Create prompt for answer generation
    prompt = f"""Based on the following context, please answer the user's query.

Context:
{context}

User Query: {query}

Provide a comprehensive and accurate answer based on the available information. If the context doesn't contain enough information, acknowledge this in your response."""
    
    # Generate answer using LLM
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info("Generated answer successfully")
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        answer = "I apologize, but I encountered an issue while generating the answer. Please try again."
    
    # Add to message history
    ai_message = AIMessage(content=answer)
    
    return {
        **state,
        "answer": answer,
        "messages": state.get("messages", []) + [ai_message]
    }


def route_next_node(state: AgentState) -> str:
    """
    Conditional edge function to determine next step in the graph.
    
    Args:
        state: Current agent state
        
    Returns:
        Name of the next node to execute
    """
    next_action = state.get("next_action")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    
    # Check if we've exceeded max iterations
    if iteration >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return "generate"
    
    # Route based on next_action
    if next_action == "retrieve":
        return "retrieve"
    elif next_action == "search_web":
        return "search_web"
    else:
        return "generate"
