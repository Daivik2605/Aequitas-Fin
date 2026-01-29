# Aequitas-Fin: AI-Powered Financial Intelligence Agent

Aequitas-Fin is a specialized RAG (Retrieval-Augmented Generation) system designed to analyze complex financial reports. By leveraging **hi_res document partitioning** and a **dual-embedding architecture**, it provides high-precision retrieval of financial data from dense PDFs, such as CIBC Annual Reports.

## 🚀 Iteration Log

> **Iterative Development Note**: This project is built in stages. Each milestone includes a performance evaluation to justify architectural choices and track retrieval accuracy.

### Milestone 1: High-Res Ingestion & Named Vector Architecture

* **Vector Database**: Successfully deployed Qdrant via Docker.
* **Parsing Strategy**: Implemented `unstructured`'s `hi_res` strategy to handle complex financial layouts and tables.
* **Dual-Model Architecture**: Implemented **Named Vectors** to support model-agnostic retrieval.
* **Local Lane (`local_bge`)**: 384-dim vectors for low-latency, private processing on Apple Silicon (M3).
* **Cloud Lane (`openai`)**: 1536-dim vectors for industry-standard high-dimensional accuracy.



## 📊 Performance Evaluation

We benchmarked the system using the query: *"What was CIBC's net income for the full year 2025?"*

| Metric | Local BGE-Small (384-dim) | OpenAI (1536-dim) |
| --- | --- | --- |
| **Search Score** | **0.8160 (High Confidence)** | **0.0000 (Dummy State)** |
| **Top Result Type** | `CompositeElement` | `CompositeElement` |
| **Retrieval Accuracy** | Successfully identified Q4 News Release Page 1. | Identified relevant text but lacked similarity score. |
| **Observation** | The local model correctly ranked contextually relevant financial summaries at the top. | Confirmed "Named Vector" structure is ready for real API integration. |

## 🛠️ Tech Stack

* **Database**: Qdrant (Vector DB)
* **Parsing**: Unstructured.io (`hi_res` layout analysis)
* **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5) & OpenAI (text-embedding-3-small)
* **Infrastructure**: Docker, Python 3.10+, Apple Silicon (M3)

## ⚙️ Installation & Setup

### 1. Prerequisites

* Docker Desktop
* Python 3.10+
* OpenAI API Key

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Daivik2605/Aequitas-Fin.git
cd Aequitas-Fin

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Initialize Services

```bash
# Start Qdrant Vector Database
docker-compose up -d

```

### 4. Data Ingestion

Place CIBC PDFs in `data/raw/cibc/FY2025/` and run:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/ingestion/ingest_docs.py

```

### 5. Run Comparison Test

To verify the dual-model retrieval performance:

```bash
python3 tests/test_retrieval.py

```

## 📈 Roadmap

* [ ] Replace OpenAI placeholder vectors with real API calls.
* [ ] Develop the LLM reasoning layer to synthesize retrieved tables into natural language answers.
* [ ] Implement a front-end dashboard for interactive financial queries.
