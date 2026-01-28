"""
Graph nodes for the LangGraph reasoning agent.
"""
from typing import Dict, Any
from langchain.schema import HumanMessage, AIMessage
from src.reasoning.state import AgentState
from src.reasoning.tools import TavilySearchTool, LocalRetrievalTool


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
    web_keywords = ["current", "latest", "news", "today", "recent", "2024", "2025", "2026"]
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
            context_parts.append(f"{i}. {doc[:500]}...")
    
    if web_results:
        context_parts.append("\n## Web Search Results:")
        for i, result in enumerate(web_results[:3], 1):
            context_parts.append(f"{i}. {result[:500]}...")
    
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
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    # Add to message history
    ai_message = AIMessage(content=answer)
    
    return {
        **state,
        "answer": answer,
        "messages": state.get("messages", []) + [ai_message]
    }


def should_continue(state: AgentState) -> str:
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
        return "generate"
    
    # Route based on next_action
    if next_action == "retrieve":
        return "retrieve"
    elif next_action == "search_web":
        return "search_web"
    else:
        return "generate"
