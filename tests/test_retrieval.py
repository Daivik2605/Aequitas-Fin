import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from openai import OpenAI

load_dotenv()

def run_comparison_test(query_text: str):
    client = QdrantClient(host="localhost", port=6333)
    oa_client = OpenAI()
    
    # 1. Generate Query Vectors
    print(f"Question: {query_text}")
    
    # Local BGE (384-dim)
    bge_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    local_vector = next(bge_model.embed([query_text])).tolist()
    
    # OpenAI (1536-dim)
    openai_res = oa_client.embeddings.create(
        input=[query_text], 
        model="text-embedding-3-small"
    )
    cloud_vector = openai_res.data[0].embedding

    # 2. Perform Dual Search
    models_to_test = [
        ("local_bge", local_vector),
        ("openai", cloud_vector)
    ]

    for model_name, vector in models_to_test:
        print(f"\n--- Results for {model_name.upper()} ---")
        search_result = client.query_points(
            collection_name="cibc_reports",
            query=vector,
            using=model_name, # Critical: Tells Qdrant which index to use
            limit=3
        ).points

        for i, hit in enumerate(search_result):
            payload = hit.payload
            meta = payload.get('metadata', {})
            print(f"{i+1}. [Score: {hit.score:.4f}] Type: {meta.get('element_type')}")
            print(f"   Source: {meta.get('source')} (Page {meta.get('page_number')})")
            print(f"   Excerpt: {payload.get('text')[:150]}...\n")

if __name__ == "__main__":
    run_comparison_test("What was CIBC's net income for the full year 2025?")