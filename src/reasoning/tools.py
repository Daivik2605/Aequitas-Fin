"""
Tools for the reasoning agent: Tavily web search and local RAG retrieval.
"""
import logging
from typing import List, Optional
from tavily import TavilyClient
from langchain.tools import Tool

logger = logging.getLogger(__name__)


class TavilySearchTool:
    """
    Tavily web search tool for retrieving up-to-date information from the internet.
    
    Tavily is a search API optimized for LLMs and RAG applications.
    """
    
    def __init__(self, api_key: str, max_results: int = 5):
        """
        Initialize Tavily search tool.
        
        Args:
            api_key: Tavily API key
            max_results: Maximum number of search results to return
        
        Raises:
            ValueError: If API key is empty or invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("Tavily API key cannot be empty")
        
        self.client = TavilyClient(api_key=api_key)
        self.max_results = max_results
        logger.info("Initialized Tavily search tool")
    
    def search(self, query: str) -> List[str]:
        """
        Perform web search using Tavily.
        
        Args:
            query: Search query
            
        Returns:
            List of search result snippets
        """
        try:
            response = self.client.search(
                query=query,
                max_results=self.max_results
            )
            
            results = []
            if "results" in response:
                for result in response["results"]:
                    # Extract content from each result
                    content = result.get("content", "")
                    url = result.get("url", "")
                    title = result.get("title", "")
                    
                    # Format the result
                    formatted = f"Title: {title}\nURL: {url}\n{content}"
                    results.append(formatted)
            
            logger.info(f"Tavily search returned {len(results)} results for query: {query[:50]}")
            return results
        except Exception as e:
            logger.error(f"Error during Tavily search: {e}", exc_info=True)
            return []
    
    def as_langchain_tool(self) -> Tool:
        """
        Convert to LangChain Tool format.
        
        Returns:
            Tool instance for LangChain integration
        """
        return Tool(
            name="web_search",
            func=self.search,
            description=(
                "Search the web for current information. "
                "Use this when you need up-to-date information, "
                "news, or information not in the local knowledge base."
            )
        )


class LocalRetrievalTool:
    """
    Local RAG retrieval tool for querying the vector database.
    
    This tool retrieves relevant documents from the local Qdrant database
    based on semantic similarity to the query.
    """
    
    def __init__(
        self,
        database,
        collection_name: str,
        embeddings,
        top_k: int = 5
    ):
        """
        Initialize local retrieval tool.
        
        Args:
            database: QdrantDatabase instance
            collection_name: Name of the collection to search
            embeddings: Embedding model for query vectorization
            top_k: Number of top results to retrieve
        """
        self.database = database
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.top_k = top_k
        logger.info(f"Initialized local retrieval tool for collection: {collection_name}")
    
    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant documents from local database.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved document contents
        """
        try:
            # Check if collection exists
            if not self.database.collection_exists(self.collection_name):
                logger.warning(f"Collection {self.collection_name} does not exist")
                return []
            
            # Embed the query
            query_vector = self.embeddings.embed_query(query)
            
            # Search the database
            results = self.database.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k
            )
            
            # Extract document contents from results
            documents = []
            for result in results:
                if hasattr(result, "payload") and "text" in result.payload:
                    documents.append(result.payload["text"])
                elif hasattr(result, "payload") and "content" in result.payload:
                    documents.append(result.payload["content"])
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}")
            return documents
        except Exception as e:
            logger.error(f"Error during local retrieval: {e}", exc_info=True)
            return []
    
    def as_langchain_tool(self) -> Tool:
        """
        Convert to LangChain Tool format.
        
        Returns:
            Tool instance for LangChain integration
        """
        return Tool(
            name="local_retrieval",
            func=self.retrieve,
            description=(
                "Retrieve relevant information from the local knowledge base. "
                "Use this for domain-specific information, internal documents, "
                "or previously ingested data."
            )
        )


def create_tools(
    tavily_api_key: Optional[str],
    database=None,
    collection_name: str = "documents",
    embeddings=None
) -> List[Tool]:
    """
    Factory function to create all tools for the agent.
    
    Args:
        tavily_api_key: API key for Tavily (None to skip web search)
        database: QdrantDatabase instance (None to skip local retrieval)
        collection_name: Collection name for retrieval
        embeddings: Embedding model for retrieval
        
    Returns:
        List of Tool instances
    """
    tools = []
    
    # Add Tavily search tool if API key provided
    if tavily_api_key:
        try:
            tavily_tool = TavilySearchTool(api_key=tavily_api_key)
            tools.append(tavily_tool.as_langchain_tool())
        except ValueError as e:
            logger.warning(f"Skipping Tavily tool: {e}")
    
    # Add local retrieval tool if database provided
    if database and embeddings:
        retrieval_tool = LocalRetrievalTool(
            database=database,
            collection_name=collection_name,
            embeddings=embeddings
        )
        tools.append(retrieval_tool.as_langchain_tool())
    
    return tools
