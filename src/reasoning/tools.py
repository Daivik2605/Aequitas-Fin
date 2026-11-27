"""
Tools for the reasoning agent: Tavily web search and local RAG retrieval.
"""
import logging
from typing import List, Optional
from tavily import TavilyClient
from langchain.tools import Tool
from config.settings import settings
from src.reasoning.state import RetrievedChunk

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
        top_k: int = 5,
        score_threshold: float = 0.3
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
        self.score_threshold = score_threshold
        self._verify_collection()
        logger.info(f"Initialized local retrieval tool for collection: {collection_name}")

    def _verify_collection(self) -> None:
        """
        Fail at construction, not at query time.

        A missing collection is a configuration error, and the old behaviour —
        log a warning and return [] — is indistinguishable from "your question
        had no good matches". That turns a one-character typo in .env into an
        afternoon spent debugging embeddings.

        Queries the client directly rather than going through
        database.collection_exists(), which catches exceptions and returns
        False. Using it here would report a missing collection when Qdrant is
        simply not running — a confidently wrong error message, which is worse
        than no error message.
        """
        client = self.database.get_client()
        existing = {c.name for c in client.get_collections().collections}
        if self.collection_name not in existing:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' does not exist. "
                f"Collections present: {sorted(existing) or 'none'}. "
                f"Run ingestion first, or check QDRANT_COLLECTION in .env"
            )
    
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Retrieve relevant documents from local database.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved document contents
        """
        try:
            # Existence is guaranteed at construction (_verify_collection), so
            # there is no per-query check here. Anything that fails below is a
            # runtime fault, not a misconfiguration.
            # Embed the query
            query_vector = self.embeddings.embed_query(query)
            
            # Search the database
            results = self.database.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k,
                score_threshold=self.score_threshold
            )
            
            # Extract document contents from results
            documents: List[RetrievedChunk] = []
            for result in results:
                if hasattr(result, "payload") and "text" in result.payload:
                    metadata = result.payload.get("metadata", {})
                    chunk: RetrievedChunk = {
                        "text": result.payload["text"],
                        "source": metadata.get("source", ""),
                        "company": metadata.get("company", ""),
                        "fiscal_year": metadata.get("fiscal_year", ""),
                        "page_number": metadata.get("page_number"),
                        "score": result.score,
                    }
                    documents.append(chunk)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}")
            return documents
        except Exception as e:
            # Degrade rather than crash the agent loop, but make the log say
            # RETRIEVAL FAILED so an empty result set is never mistaken for
            # "nothing matched".
            logger.error(f"RETRIEVAL FAILED for query {query[:50]!r}: {e}", exc_info=True)
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
            embeddings=embeddings,
            top_k=settings.RETRIEVAL_TOP_K,
            score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD
        )
        tools.append(retrieval_tool.as_langchain_tool())
    
    return tools
