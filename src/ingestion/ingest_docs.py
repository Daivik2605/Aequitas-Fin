import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from fastembed import TextEmbedding
from qdrant_client import models
from src.core.database import QdrantDatabase

load_dotenv()

def process_cibc_reports(data_dir: str):
    # 1. Setup Database and Model
    db = QdrantDatabase(host="localhost", port=6333)
    client = db.get_client()
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 2. Recreate Collection with Named Vectors
    print("🗑️ Resetting Qdrant collection for Named Vectors...")
    client.recreate_collection(
        collection_name="cibc_reports",
        vectors_config={
            "local_bge": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "openai": models.VectorParams(size=1536, distance=models.Distance.COSINE),
        },
    )

    raw_path = Path(data_dir)
    # Recursively find all PDFs (e.g., in FY2025/)
    pdf_files = list(raw_path.glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDFs found in {data_dir}. Check your folders!")
        return

    for pdf_file in pdf_files:
        print(f"\n🔄 Partitioning: {pdf_file.name} using hi_res...")
        
        # 3. High-Resolution Layout Analysis
        elements = partition_pdf(
            filename=str(pdf_file),
            strategy="hi_res",
            hi_res_model_name="yolox", # Optimized for M3/CPU speed
            chunking_strategy="by_title",
            max_characters=2000,
        )

        # 4. Batch Embedding (Speed Hack)
        texts = [el.text for el in elements if el.text.strip()]
        print(f"🧠 Generating BGE embeddings for {len(texts)} chunks...")
        bge_embeddings = list(embedding_model.embed(texts))
        
        # Metadata logic
        year = pdf_file.parent.name.replace("FY", "")
        points = []

        # 5. Build Qdrant Points
        for i, (element, bge_vec) in enumerate(zip(elements, bge_embeddings)):
            el_dict = element.to_dict()
            metadata = el_dict.get("metadata", {})
            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "local_bge": bge_vec.tolist(),
                    "openai": [0.0] * 1536, # Placeholder
                },
                payload={
                    "text": element.text,
                    "metadata": {
                        "source": pdf_file.name,
                        "fiscal_year": year,
                        "element_type": el_dict.get("type"),
                        "page_number": metadata.get("page_number")
                    }
                }
            ))

        # 6. Upload
        client.upsert(collection_name="cibc_reports", points=points)
        print(f"✅ Successfully indexed {len(points)} points from {pdf_file.name}")

if __name__ == "__main__":
    process_cibc_reports("data/raw/cibc")