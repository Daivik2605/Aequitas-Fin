"""
Database module for Qdrant vector store connection and operations.
"""
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


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
        elif url:
            # Use cloud instance
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            # Use host:port connection
            self.client = QdrantClient(host=host, port=port)
    
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
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
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
        except Exception:
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
            return True
        except Exception as e:
            print(f"Error upserting vectors: {e}")
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
            return results
        except Exception as e:
            print(f"Error searching vectors: {e}")
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
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def get_client(self) -> QdrantClient:
        """
        Get the underlying Qdrant client instance.
        
        Returns:
            QdrantClient instance
        """
        return self.client
