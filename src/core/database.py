"""
Database module for Qdrant vector store connection and operations.
"""
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


class QdrantDatabase:
    """
    Qdrant client connection class for managing vector database operations.
    
    This class provides a wrapper around the Qdrant client for:
    - Establishing connections to Qdrant (local or remote)
    - Creating and managing collections
    - Storing and retrieving vectors
    - Performing similarity searches
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None
    ):
        """
        Initialize Qdrant client connection.
        
        Args:
            host: Qdrant server host (default: localhost)
            port: Qdrant server port (default: 6333)
            url: Full URL for Qdrant cloud instance
            api_key: API key for Qdrant cloud
            path: Path for local persistent storage
        """
        if path:
            # Use local persistent storage
            self.client = QdrantClient(path=path)
            logger.info(f"Initialized Qdrant client with local storage: {path}")
        elif url:
            # Use cloud instance
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"Initialized Qdrant client with URL: {url}")
        else:
            # Use host:port connection
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Initialized Qdrant client at {host}:{port}")
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            distance: Distance metric (COSINE, EUCLID, DOT)
            
        Returns:
            True if collection created successfully
        """
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists
        """
        try:
            collections = self.client.get_collections().collections
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def upsert_vectors(
        self,
        collection_name: str,
        points: List[PointStruct]
    ) -> bool:
        """
        Insert or update vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of PointStruct objects containing vectors and payloads
            
        Returns:
            True if operation successful
        """
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} vectors to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors to {collection_name}: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            logger.debug(f"Search returned {len(results)} results from {collection_name}")
            return results
        except Exception as e:
            logger.error(f"Error searching {collection_name}: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deletion successful
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def get_client(self) -> QdrantClient:
        """
        Get the underlying Qdrant client instance.
        
        Returns:
            QdrantClient instance
        """
        return self.client
