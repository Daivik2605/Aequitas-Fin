"""
LangGraph state machine for the reasoning agent.

This module builds a graph-based workflow with nodes for:
- Routing decisions
- RAG retrieval
- Web search
- Answer generation
"""
import logging
from typing import Optional
from langgraph.graph import StateGraph, END
from src.reasoning.state import AgentState, create_initial_state
from src.reasoning.nodes import (
    router_node,
    rag_retrieval_node,
    web_search_node,
    generate_answer_node,
    route_next_node
)
from src.reasoning.tools import TavilySearchTool, LocalRetrievalTool

logger = logging.getLogger(__name__)


class ReasoningGraph:
    """
    LangGraph-based reasoning agent with RAG and Web Search capabilities.
    
    The graph follows this flow:
    1. Router: Decides whether to use RAG, web search, or generate answer
    2. RAG Node: Retrieves from local database (if needed)
    3. Web Search Node: Searches the web (if needed)
    4. Generate Node: Produces final answer
    """
    
    def __init__(
        self,
        llm,
        retrieval_tool: Optional[LocalRetrievalTool] = None,
        search_tool: Optional[TavilySearchTool] = None
    ):
        """
        Initialize the reasoning graph.
        
        Args:
            llm: Language model for answer generation
            retrieval_tool: Optional local retrieval tool
            search_tool: Optional web search tool
        """
        self.llm = llm
        self.retrieval_tool = retrieval_tool
        self.search_tool = search_tool
        self.graph = self._build_graph()
        logger.info("Reasoning graph initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Returns:
            Compiled StateGraph instance
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", router_node)
        
        # Add RAG retrieval node if tool is available
        if self.retrieval_tool:
            workflow.add_node(
                "retrieve",
                lambda state: rag_retrieval_node(state, self.retrieval_tool)
            )
            logger.debug("Added RAG retrieval node")
        else:
            logger.warning("RAG retrieval tool not available - retrieval disabled")
        
        # Add web search node if tool is available
        if self.search_tool:
            workflow.add_node(
                "search_web",
                lambda state: web_search_node(state, self.search_tool)
            )
            logger.debug("Added web search node")
        else:
            logger.warning("Web search tool not available - web search disabled")
        
        # Add answer generation node
        workflow.add_node(
            "generate",
            lambda state: generate_answer_node(state, self.llm)
        )
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            route_next_node,
            {
                "retrieve": "retrieve" if self.retrieval_tool else "generate",
                "search_web": "search_web" if self.search_tool else "generate",
                "generate": "generate"
            }
        )
        
        # Add edges back to router from retrieval and search nodes
        if self.retrieval_tool:
            workflow.add_edge("retrieve", "router")
        
        if self.search_tool:
            workflow.add_edge("search_web", "router")
        
        # Add edge from generate to END
        workflow.add_edge("generate", END)
        
        # Compile the graph
        return workflow.compile()
    
    def run(self, query: str, max_iterations: int = 5) -> dict:
        """
        Run the reasoning graph on a query.
        
        Args:
            query: User's query
            max_iterations: Maximum number of iterations
            
        Returns:
            Final state with answer
        """
        # Create initial state
        initial_state = create_initial_state(query, max_iterations)
        
        logger.info(f"Running graph for query: {query[:50]}")
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return result
    
    async def arun(self, query: str, max_iterations: int = 5) -> dict:
        """
        Asynchronously run the reasoning graph on a query.
        
        Args:
            query: User's query
            max_iterations: Maximum number of iterations
            
        Returns:
            Final state with answer
        """
        # Create initial state
        initial_state = create_initial_state(query, max_iterations)
        
        logger.info(f"Running graph async for query: {query[:50]}")
        
        # Run the graph asynchronously
        result = await self.graph.ainvoke(initial_state)
        
        return result
    
    def stream(self, query: str, max_iterations: int = 5):
        """
        Stream the reasoning graph execution.
        
        Args:
            query: User's query
            max_iterations: Maximum number of iterations
            
        Yields:
            State updates as the graph executes
        """
        # Create initial state
        initial_state = create_initial_state(query, max_iterations)
        
        logger.info(f"Streaming graph for query: {query[:50]}")
        
        # Stream the graph execution
        for state in self.graph.stream(initial_state):
            yield state


def create_reasoning_graph(
    llm,
    retrieval_tool: Optional[LocalRetrievalTool] = None,
    search_tool: Optional[TavilySearchTool] = None
) -> ReasoningGraph:
    """
    Factory function to create a reasoning graph.
    
    Args:
        llm: Language model instance
        retrieval_tool: Optional local retrieval tool
        search_tool: Optional web search tool
        
    Returns:
        ReasoningGraph instance
    """
    return ReasoningGraph(
        llm=llm,
        retrieval_tool=retrieval_tool,
        search_tool=search_tool
    )
