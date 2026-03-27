"""
Embedding wrapper for local BGE model via FastEmbed.
"""
from fastembed import TextEmbedding


class FastEmbedWrapper:
    """
    Wraps BAAI/bge-small-en-v1.5 via fastembed for query embedding.

    embed_query returns a named-vector tuple ("local_bge", list[float])
    so it can be passed directly to QdrantDatabase.search() when the
    collection uses named vectors.
    """

    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_query(self, text: str) -> tuple:
        """
        Embed a single query string.

        Returns:
            Tuple of (vector_name, embedding) ready for Qdrant named-vector search.
        """
        embeddings = list(self.model.embed([text]))
        return ("local_bge", embeddings[0].tolist())
