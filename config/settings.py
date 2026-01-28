"""
Configuration settings for Aequitas-Fin agent.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and configuration."""
    
    # Qdrant Database Settings
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "aequitas_documents")
    
    # LLM Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")
    MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "2048"))
    
    # Tavily Search Settings
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    
    # Retrieval Settings
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
    
    # Agent Settings
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate critical settings.
        
        Returns:
            True if settings are valid
        """
        # Check if Tavily API key is set (optional but recommended)
        if not cls.TAVILY_API_KEY:
            print("Warning: TAVILY_API_KEY not set. Web search will not be available.")
        
        return True


# Create global settings instance
settings = Settings()
