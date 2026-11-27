"""
Query-side embedding wrappers.

Each wrapper embeds a QUERY and returns a (vector_name, vector) tuple ready
for Qdrant named-vector search, so the caller can swap retrieval lanes
without knowing anything about the underlying model.

ASYMMETRIC RETRIEVAL
RAG is an asymmetric task: a short question is matched against a long
passage that ANSWERS it, not one that RESEMBLES it. BGE v1.5 models are
trained for this by prefixing the query side with an instruction during
contrastive fine-tuning; passages are embedded plain. Omitting the
instruction embeds a question as though it were a document, which pushes it
toward text that looks like a question instead of text that answers one.

OpenAI's text-embedding-3 models need no such prefix — they are trained to
handle both roles — which is why only the BGE lane has this switch.

The switch exists so the instruction's effect can be MEASURED rather than
assumed: run the eval set with it on and off and report the difference.
Whatever is used in evaluation must also be used in the app, or the measured
numbers do not describe the shipped system.
"""

from fastembed import TextEmbedding

# BAAI's documented query instruction for bge-*-en-v1.5 retrieval.
# Applied explicitly rather than relying on fastembed's query_embed() so the
# exact string is visible, version-stable, and quotable in the write-up.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class FastEmbedWrapper:
    """
    Local lane: BAAI/bge-small-en-v1.5 (384-dim) via fastembed.

    Free, private, CPU-friendly. Caps at 512 input tokens and truncates
    silently beyond that — not a concern for short queries, but it is for
    the long chunks embedded at ingestion time.
    """

    def __init__(self, use_query_instruction: bool = True):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.use_query_instruction = use_query_instruction

    def embed_query(self, text: str) -> tuple:
        """
        Embed a single query string.

        Returns:
            Tuple of (vector_name, embedding) ready for Qdrant named-vector search.
        """
        payload = (
            BGE_QUERY_INSTRUCTION + text if self.use_query_instruction else text
        )
        embedding = next(iter(self.model.embed([payload])))
        return ("local_bge", embedding.tolist())


class OpenAIEmbedWrapper:
    """
    Cloud lane: text-embedding-3-small (1536-dim).

    No query instruction — the model is trained to handle queries and
    passages in the same space without a mode switch.
    """

    def __init__(self, api_key: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def embed_query(self, text: str) -> tuple:
        """
        Embed a single query string.

        Returns:
            Tuple of (vector_name, embedding) ready for Qdrant named-vector search.
        """
        response = self.client.embeddings.create(input=[text], model="text-embedding-3-small")
        embedding = response.data[0].embedding
        return ("openai", embedding)
