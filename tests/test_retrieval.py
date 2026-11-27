"""
Manual dual-lane retrieval comparison.

NOT a pytest test — it prints results for a human to read. Turning it into
an assertion-bearing test requires a ground truth set (Stage C). It is kept
because reading real retrieved chunks catches problems that aggregate
metrics hide: a plausible-looking score on a chunk that is missing its
table header, for instance.

Two things changed from the original:
  * collection, host and port come from settings instead of literals, so it
    can never drift from what ingestion actually wrote to;
  * it embeds through the SAME wrapper classes the application uses, so it
    measures what the app does — including the BGE query instruction. A
    comparison script that embeds differently from the app is measuring a
    system nobody ships.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from config.settings import settings
from src.core.embeddings import FastEmbedWrapper, OpenAIEmbedWrapper

load_dotenv()


def run_comparison_test(query_text: str, limit: int = 3):
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    # Fail loudly. A missing collection returning zero results looks exactly
    # like a query with no good matches, which is how a config typo turns
    # into an afternoon of debugging the embeddings.
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION not in existing:
        raise SystemExit(
            f"Collection '{settings.QDRANT_COLLECTION}' does not exist.\n"
            f"Collections present: {sorted(existing) or 'none'}\n"
            f"Run ingestion first, or check QDRANT_COLLECTION in .env"
        )

    print(f"Question:   {query_text}")
    print(f"Collection: {settings.QDRANT_COLLECTION}")

    lanes = [FastEmbedWrapper().embed_query(query_text)]

    if os.getenv("OPENAI_API_KEY"):
        lanes.append(OpenAIEmbedWrapper(os.getenv("OPENAI_API_KEY")).embed_query(query_text))
    else:
        print("\nOPENAI_API_KEY not set - skipping the cloud lane")

    for vector_name, vector in lanes:
        print(f"\n--- Results for {vector_name.upper()} ---")
        hits = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=vector,
            using=vector_name,  # selects which named vector index to search
            limit=limit,
        ).points

        if not hits:
            print("  (no results)")
            continue

        for i, hit in enumerate(hits, 1):
            payload = hit.payload or {}
            meta = payload.get("metadata", {})
            print(f"{i}. [score {hit.score:.4f}] {meta.get('element_type')} "
                  f"| {meta.get('token_count')} tok")
            print(f"   {meta.get('source')} (page {meta.get('page_number')})")
            excerpt = (payload.get("text") or "")[:200].replace("\n", " / ")
            print(f"   {excerpt}...\n")


if __name__ == "__main__":
    run_comparison_test("What was CIBC's net income for fiscal 2024?")
