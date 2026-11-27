# Aequitas-Fin — Project Source of Truth

> **Audit date:** 2026-07-11
> **Branch audited:** `rebuild/architecture-audit` (HEAD `6ac36e5`)
> **Method:** Every claim in this document was checked against the actual code, configuration, and on-disk state of this repository. Where the repository cannot prove something, it is marked **Unknown / needs verification**. Nothing here is copied from the README without verification.

---

## 1. Project overview

**What it is.** Aequitas-Fin is a prototype Retrieval-Augmented Generation (RAG) system for answering questions about company financial reports (PDF annual reports, news releases, MD&A documents, investor presentations). The only corpus used so far is CIBC (Canadian Imperial Bank of Commerce) filings.

**The problem it is intended to solve.** Financial reports are long, dense PDFs full of tables. Finding a specific figure ("What was net income in FY2024?") by hand is slow. Aequitas-Fin parses these PDFs, stores the text as searchable vectors in a database, and lets a user ask questions in plain English. A local language model then writes an answer based on the retrieved passages.

**Who would use it.** Today: only the project owner, on their own machine, through a terminal. In the future: analysts, students, or developers who want to query financial filings without reading them end to end.

**Why the project exists.** It is a learning/portfolio project built in iterations (the README calls this out explicitly as an "Iteration Log"). Each stage adds one piece of a production RAG system: ingestion, vector storage, embedding comparison, and a reasoning loop. It is not a product and is not production-ready.

---

## 2. Project goals

### Current prototype goals (what the code is trying to do right now)
- Parse CIBC PDF reports with layout-aware parsing (`unstructured` `hi_res`) — implemented in `src/ingestion/ingest_docs.py`.
- Store each chunk with **two** embeddings (local BGE and OpenAI) side by side in Qdrant "named vectors", so the two models can be compared on the same data — implemented.
- Run an interactive terminal Q&A loop where a local Ollama LLM answers from retrieved chunks — implemented in `main.py` + `src/reasoning/`.
- Compare retrieval quality of the two embedding models on a sample query — a manual script exists (`tests/test_retrieval.py`).

### Near-term goals (started or clearly implied, not done)
- Reorganize the corpus as `data/raw/{company}/FY{year}/*.pdf` — the folder structure already follows this, and ingestion metadata code assumes it, but ingestion is still hardcoded to the `cibc` folder.
- Real retrieval evaluation (the `src/evaluation/` package exists but both files are **empty**).
- Real automated tests (pytest currently collects **zero** tests — verified by running `pytest --collect-only`).
- README roadmap items: LLM reasoning layer (now partially done, README not updated) and replacing OpenAI placeholder vectors with real API calls (done in commit `cb4300b`, README still lists it as open).

### Long-term goals (stated in README roadmap or implied, no code exists)
- A front-end dashboard for interactive financial queries.
- Multi-company, multi-year analysis.
- Observability (a Phoenix container is declared in `docker-compose.yml` but nothing in the code ever talks to it).

None of the near-term or long-term goals should be described as completed.

---

## 3. What the project can do today

Status labels used below:
- **Implemented and verified** — code exists and this audit could confirm it works (or on-disk artifacts prove it ran).
- **Implemented but not verified** — code exists and looks complete, but nothing in the repo proves it runs successfully.
- **Partially implemented** — some of the pieces exist.
- **Prototype only** — works only as a one-off manual script.
- **Planned** — no code exists.

| Capability | Status | Evidence |
|---|---|---|
| Parse PDFs with `unstructured` `hi_res` and chunk by title | **Implemented and verified** | `src/ingestion/ingest_docs.py:50-56` (`partition_pdf`, `strategy="hi_res"`, `chunking_strategy="by_title"`). Two populated Qdrant collections exist on disk (`qdrant_storage/collections/cibc_reports/` 23 MB, `financial_reports/` 21 MB), proving ingestion ran. |
| Dual embeddings (BGE 384-dim + OpenAI 1536-dim) stored as Qdrant named vectors | **Implemented and verified** | `src/ingestion/ingest_docs.py:30-36` creates named vectors `local_bge` and `openai`; both on-disk collection `config.json` files show exactly this schema. |
| Real OpenAI embedding calls (`text-embedding-3-small`) during ingestion | **Implemented but not verified** | `get_openai_embeddings()` in `src/ingestion/ingest_docs.py:13-20`. Commit `cb4300b` replaced the old zero-vector placeholder. Whether the current on-disk `financial_reports` collection contains real (non-zero) OpenAI vectors is **Unknown / needs verification** (requires querying a running Qdrant). |
| Company/fiscal-year metadata on chunks | **Implemented but not verified** | `src/ingestion/ingest_docs.py:65-69` derives `company` and `fiscal_year` from the folder path. Whether the currently stored points carry this metadata depends on which ingestion version produced them — **Unknown / needs verification**. |
| Semantic search over stored chunks (BGE lane) | **Implemented but not verified** | `QdrantDatabase.search()` (`src/core/database.py:125-164`) + `LocalRetrievalTool.retrieve()` (`src/reasoning/tools.py:119-157`) + `FastEmbedWrapper.embed_query()` (`src/core/embeddings.py:19-27`). No test or log in the repo proves an end-to-end successful run. |
| Interactive terminal Q&A with a local Ollama LLM | **Implemented but not verified** | `main.py:21-134`; `LLMModels.get_default_model()` (`src/core/models.py:99-115`). Requires a locally running Ollama with the model pulled. |
| LangGraph routing workflow (retrieve → generate, optional web search) | **Implemented but not verified** | `src/reasoning/graph.py` (`ReasoningGraph._build_graph`), `src/reasoning/nodes.py` (`router_node`, `rag_retrieval_node`, `web_search_node`, `generate_answer_node`). Routing is a keyword heuristic, not an LLM decision (`nodes.py:33-51`). |
| Optional Tavily web search | **Implemented but not verified** | `TavilySearchTool` (`src/reasoning/tools.py:12-86`); enabled only if `TAVILY_API_KEY` is set (`main.py:55`). |
| CLI model override (`--model`) | **Implemented and verified** (by code inspection; trivially simple) | `main.py:22-26`, commit `65093e5`. |
| Side-by-side retrieval comparison of the two embedding models | **Prototype only** | `tests/test_retrieval.py` — a manual print script with no assertions, pointed at the **old** `cibc_reports` collection. |
| Retrieval evaluation metrics | **Planned** | `src/evaluation/metrics.py` and `src/evaluation/run_evals.py` are both **0 bytes** (empty files). |
| Automated tests | **Planned** | pytest collects zero tests (verified: `no tests collected`). |
| Citations in answers | **Planned** | Retrieval discards all metadata before the LLM sees it (`tools.py:146-151` returns only `payload["text"]`). |
| API / UI / dashboard | **Planned** | No web framework code exists anywhere in the repo. |
| Observability (Phoenix / tracing) | **Planned** | Phoenix container declared in `docker-compose.yml:16-22`; zero references in Python code (verified by grep). |

---

## 4. What the project cannot do today

Explicit list of missing or unreliable capabilities:

- **Ingestion reliability.** Ingestion is destructive and non-repeatable: every run calls `client.recreate_collection(...)` (`src/ingestion/ingest_docs.py:30`), which **deletes all previously stored data**. Point IDs are random UUIDs (`ingest_docs.py:78`), so re-ingesting the same file would duplicate content if the wipe didn't happen first. There is also an embedding/element alignment bug (see §7, failure risks).
- **Retrieval evaluation.** There is no evaluation. `src/evaluation/` contains two empty files. The only "benchmark" is one manual query printed to the terminal.
- **Answer correctness.** Nothing checks whether the LLM's answer is right. Generation runs at temperature 0.7 (`config/settings.py:23`), which is high for factual financial Q&A.
- **Citations.** Impossible today: page numbers, source filenames, and scores are stored in Qdrant payloads but stripped out before context construction (`src/reasoning/tools.py:146-151`).
- **Multi-company support.** The ingestion entry point hardcodes `process_cibc_reports("data/raw/cibc")` (`ingest_docs.py:100`) and each run wipes the collection, so a second company cannot be added without deleting the first.
- **Multi-year support.** Fiscal-year metadata is written but never filtered on at query time. A question about FY2024 can retrieve FY2025 chunks and vice versa. No Qdrant filter is ever constructed anywhere in the code.
- **Structured financial analysis.** There are no financial calculation tools of any kind (no ratio math, no table math, no number extraction). See §16.
- **API/UI.** None. The only interface is `input()` in a terminal loop (`main.py:99`).
- **Observability.** No logging to files, no tracing, no metrics, no cost tracking. Phoenix runs as a container but nothing sends data to it.
- **Deployment.** No application container exists. `.docker/Dockerfile.agent` is tracked in git but is an **empty file**. `docker-compose.yml` only runs infrastructure (Qdrant, Phoenix).
- **Tests.** Zero automated tests. `tests/test_retrieval.py` is a manual script (see §18).
- **Security.** No input sanitization, no prompt-injection defenses, document text is sent to OpenAI's API during ingestion (privacy consideration for sensitive documents).
- **Production readiness.** Not production-ready in any dimension: no error budget, no idempotency, no health checks, no CI, no pinned dependencies, and `requirements.txt` is missing three packages the code imports (see §19).

---

## 5. Current repository structure

```
Aequitas-Fin/
├── README.md                  # Project description; partially outdated (see §17)
├── main.py                    # Entry point: interactive terminal Q&A agent
├── requirements.txt           # Python deps — INCOMPLETE (missing unstructured, fastembed, openai)
├── docker-compose.yml         # Qdrant + Phoenix containers (Phoenix unused by code)
├── .env                       # Local secrets (NOT committed — verified via `git ls-files`)
├── .env.example               # Template — out of sync with code (see §19)
├── .gitignore                 # Has a stray "EOF" line (line 13); ignores data/raw, data/gold but not data/silver, data/raw_backup
├── .docker/
│   └── Dockerfile.agent       # EMPTY file, tracked in git, referenced by nothing
├── config/
│   ├── settings.py            # Settings class: reads env vars, provides defaults
│   └── prompts.py             # Prompt templates — DEAD CODE, never imported anywhere
├── src/
│   ├── core/
│   │   ├── database.py        # QdrantDatabase wrapper (connect, create, upsert, search, delete)
│   │   ├── embeddings.py      # FastEmbedWrapper: BGE query embedding, returns named-vector tuple
│   │   └── models.py          # LLMModels: ChatOllama factory; also unused EmbeddingModel class
│   ├── ingestion/
│   │   └── ingest_docs.py     # PDF → chunks → dual embeddings → Qdrant (script, run manually)
│   ├── reasoning/
│   │   ├── state.py           # AgentState TypedDict for LangGraph
│   │   ├── nodes.py           # router / retrieve / web-search / generate nodes
│   │   ├── graph.py           # ReasoningGraph: builds and compiles the LangGraph
│   │   └── tools.py           # TavilySearchTool, LocalRetrievalTool
│   └── evaluation/
│       ├── metrics.py         # EMPTY (0 bytes)
│       └── run_evals.py       # EMPTY (0 bytes)
├── tests/
│   └── test_retrieval.py      # Manual comparison script — NOT a pytest test (no assertions)
├── data/                      # Untracked; see §6
│   ├── raw/cibc/FY2024/       # 5 CIBC PDFs (current corpus)
│   ├── raw_backup/cibc/FY2025/# 4 CIBC PDFs (previous corpus)
│   ├── silver/                # empty
│   └── gold/                  # empty
├── qdrant_storage/            # Qdrant Docker volume (ignored); holds 2 collections
├── mlx_models/                # empty (ignored)
├── notebooks/                 # empty
└── docs/
    └── AEQUITAS_FIN_SOURCE_OF_TRUTH.md   # this document
```

Purpose of each important piece:

| Path | Purpose |
|---|---|
| `main.py` | Wires everything together: settings → Qdrant client → Ollama LLM → tools → LangGraph → interactive loop. |
| `config/settings.py` | The `Settings` class. Single place env vars are read (plus two scripts that read `OPENAI_API_KEY` directly). |
| `config/prompts.py` | Four prompt templates (`SYSTEM_PROMPT`, `ROUTING_PROMPT`, `ANSWER_GENERATION_PROMPT`, `QUERY_REFINEMENT_PROMPT`). **Never imported by any module** (verified by grep) — the real generation prompt is a different inline string in `src/reasoning/nodes.py:157-164`. |
| `src/core/database.py` | `QdrantDatabase` class. Note: ingestion **bypasses** this wrapper's `create_collection`/`upsert_vectors` methods and uses the raw client via `db.get_client()` instead. Only `search()` and `collection_exists()` are used by the query path. |
| `src/core/models.py` | `LLMModels` (Ollama chat model factory — `get_default_model()` is the only method actually used), plus dead code: `get_llama3`, `get_mistral`, `get_llm`, and an `EmbeddingModel` class that duplicates `FastEmbedWrapper`. The module docstring still says "placeholder implementations", which is outdated — it is a real Ollama wrapper. |
| `src/ingestion/ingest_docs.py` | The whole ingestion pipeline in one function, `process_cibc_reports()`. |
| `src/reasoning/*` | The LangGraph workflow (see §15). |
| `tests/test_retrieval.py` | One function, `run_comparison_test()`, run via `python tests/test_retrieval.py`. |

---

## 6. Current data organization

**Where raw PDFs live (actual, verified on disk):**

```
data/raw/cibc/FY2024/
├── annual_report.pdf
├── inventorPresentation2024.pdf     # note the typo in the filename ("inventor")
├── mda.pdf
├── news_release.pdf
└── supplementary_financial_information.pdf

data/raw_backup/cibc/FY2025/
├── mda-2025-en.pdf
├── q425financials-en.pdf
├── q425newsrelease-en.pdf
└── q425presentation-en.pdf
```

**Intended path convention:** `data/raw/{company}/FY{year}/*.pdf`. The current layout follows it.

**How company and fiscal year are inferred:** from the folder path, in `src/ingestion/ingest_docs.py:65-69`:
- `fiscal_year` = name of the PDF's parent folder with `FY` stripped (e.g. `FY2024` → `"2024"`).
- `company` = name of the grandparent folder, lowercased (e.g. `cibc`).

This only works if PDFs sit **exactly** at depth `company/FYyyyy/file.pdf`. Because ingestion uses a recursive glob (`raw_path.glob("**/*.pdf")`, `ingest_docs.py:40`), a PDF placed directly in `data/raw/cibc/` would silently get `fiscal_year="cibc"` and `company="raw"` — wrong metadata, no error.

**Mismatches between the data layout and the code/docs:**
1. `README.md:71` tells users to place PDFs in `data/raw/cibc/FY2025/`, but the live corpus is `FY2024` and the FY2025 set was moved to `data/raw_backup/`. The README instruction is stale.
2. The ingestion entry point (`ingest_docs.py:100`) hardcodes `data/raw/cibc` — the `{company}` part of the convention is not yet parameterized.
3. Which fiscal year the on-disk Qdrant collections actually contain is **Unknown / needs verification** (the two collections were populated at different times, likely from different corpora).

**Where benchmark data should live:** nowhere is designated yet. `data/gold/` exists and is empty — its name suggests it is intended for gold-standard (ground-truth) evaluation data, and `.gitignore` already lists it. `data/silver/` also exists, empty, presumably for intermediate/processed output (a "medallion" layout). Neither is referenced by any code.

**Local/generated data:** `qdrant_storage/` (the Qdrant Docker volume), `mlx_models/` (empty; ignored so model weights are never committed).

**Git hygiene — what is committed vs. ignored (verified via `git ls-files` and `.gitignore`):**
- Committed: all source, config, README, `docker-compose.yml`, `.env.example`.
- Ignored: `.env`, `venv/`, `qdrant_storage/`, `data/raw/`, `data/gold/`, `mlx_models/`, `.claude/`.
- **Gap:** `data/silver/`, `data/raw_backup/`, and `data/.DS_Store` are **not ignored and not committed** — `git status` shows `?? data/`. Either ignore all of `data/` (except perhaps a committed gold benchmark) or commit intentionally.
- **Bug:** `.gitignore:13` contains a stray literal line `EOF` — leftover from a heredoc paste. Harmless but should be removed.

---

## 7. Document ingestion pipeline

Everything happens in one function: `process_cibc_reports(data_dir)` in `src/ingestion/ingest_docs.py`. It is run manually: `python src/ingestion/ingest_docs.py` (with `PYTHONPATH=.`). Exact flow from PDF to stored Qdrant point:

**Step 0 — Setup** (`ingest_docs.py:23-26`)
- Input: nothing.
- Processing: connects to Qdrant at hardcoded `localhost:6333` (ignores `settings.QDRANT_HOST/PORT`); loads `BAAI/bge-small-en-v1.5` via `fastembed.TextEmbedding`.
- Failure risks: Qdrant not running → connection error; first run downloads the BGE model (needs network).

**Step 1 — Collection reset** (`ingest_docs.py:29-36`)
- Processing: `client.recreate_collection("financial_reports", vectors_config={"local_bge": 384-dim cosine, "openai": 1536-dim cosine})`.
- Output: a fresh, **empty** collection with two named vector spaces.
- Failure risks: **destroys all existing data every run**. `recreate_collection` is also deprecated in recent `qdrant-client` versions.

**Step 2 — File discovery** (`ingest_docs.py:38-44`)
- Input: `data_dir` (hardcoded to `"data/raw/cibc"` at `ingest_docs.py:100`).
- Processing: recursive glob `**/*.pdf`.
- Output: list of PDF paths. Prints an error and returns if none found.
- Failure risks: path is relative, so the script only works when run from the repo root.

**Step 3 — Partitioning + chunking** (`ingest_docs.py:50-56`)
- Input: one PDF path.
- Processing: `partition_pdf(filename=..., strategy="hi_res", hi_res_model_name="yolox", chunking_strategy="by_title", max_characters=2000)`. Because `chunking_strategy` is passed to `partition_pdf`, Unstructured partitions **and** chunks in one call; the returned `elements` are already chunks (mostly `CompositeElement` objects).
- Output: a list of chunk elements, each with `.text` and `.metadata` (page number, type, etc.).
- Failure risks: `hi_res` is slow and depends on ML models (YOLOX layout detection) and system libraries (poppler, tesseract) that are not declared anywhere; a scanned/encrypted PDF could fail or produce garbage; there is no try/except, so one bad PDF kills the whole run after the collection was already wiped.

**Step 4 — Embedding** (`ingest_docs.py:58-63`)
- Input: `texts = [el.text for el in elements if el.text.strip()]` — chunk texts, **excluding** empty/whitespace ones.
- Processing: BGE embeds all texts locally; `get_openai_embeddings()` (`ingest_docs.py:13-20`) sends the texts to OpenAI's `text-embedding-3-small` in batches of 100.
- Output: two aligned lists of vectors (aligned with `texts`, **not** with `elements` — this matters, see the bug below).
- Failure risks: needs `OPENAI_API_KEY`; network/rate-limit failures abort the run; API cost scales with corpus size.

**Step 5 — Point construction** (`ingest_docs.py:65-93`)
- Input: elements + both embedding lists + folder-derived metadata.
- Processing: for each element, builds a `PointStruct` with `id=str(uuid.uuid4())`, `vector={"local_bge": ..., "openai": ...}`, and payload `{"text", "metadata": {"source", "company", "fiscal_year", "element_type", "page_number"}}`.
- Output: list of points for this PDF.
- **Confirmed alignment bug:** the loop is `zip(elements, bge_embeddings, openai_embeddings)` (`ingest_docs.py:73`), but the embeddings were computed from the **filtered** `texts` list. If any element had empty text, every element after it is paired with the **wrong vector** (off-by-one shift), and `zip` silently drops the tail. Chunks would then be retrieved by similarity to a *different* chunk's text. Whether the current corpus triggered this depends on whether any element had blank text — **Unknown / needs verification**, but the bug is real in the code.

**Step 6 — Upload** (`ingest_docs.py:96`)
- Processing: `client.upsert(collection_name="financial_reports", points=points)` — one upsert per PDF.
- Failure risks: since IDs are random UUIDs, "upsert" never updates — it only inserts. Idempotency comes solely from the destructive wipe in Step 1.

**Where the specific items you asked about live:**

| Item | Present? | Location |
|---|---|---|
| `partition_pdf` | Yes | `src/ingestion/ingest_docs.py:5` (import), `:50` (call) |
| `strategy="hi_res"` | Yes | `ingest_docs.py:52` |
| `infer_table_structure` | **No — not present anywhere in the code** (verified by grep). Tables are not extracted as structured HTML; table text ends up as plain chunk text. | — |
| `chunk_by_title` | Present as the parameter `chunking_strategy="by_title"` (`ingest_docs.py:54`); the standalone `chunk_by_title()` function is **not** imported. | `ingest_docs.py:54` |
| `CompositeElement` | **Not referenced in code.** It appears only in `README.md:26-27` as the observed element type of retrieval results (which is consistent with by-title chunking output). | `README.md` only |
| OCR / table extraction options | **None configured.** No `ocr_languages`, `languages`, `extract_images`, or table options are passed. Whatever OCR happens is Unstructured's internal default behavior for `hi_res`. | — |

---

## 8. What Unstructured.io is doing

In plain ETL terms, for this project:

**Extract** — `partition_pdf` opens each PDF and, because `strategy="hi_res"` is set, runs a layout-detection model (`yolox`) over each page image to find where titles, paragraphs, and tables physically sit on the page. This is what lets it handle dense financial layouts better than plain text extraction.

**Transform** — Unstructured labels each detected region with a type (Title, NarrativeText, Table, etc.) and attaches metadata such as the page number. Because the code also passes `chunking_strategy="by_title"` with `max_characters=2000`, Unstructured then merges consecutive elements into chunks that start at section titles, producing `CompositeElement` chunks up to 2,000 characters.

**Load** — **Unstructured does NOT load anything.** Loading into Qdrant is done entirely by this project's own code: point construction at `ingest_docs.py:73-93` and `client.upsert(...)` at `ingest_docs.py:96`. Unstructured's job ends when it returns the chunk list.

**Actual configuration from the code** (`src/ingestion/ingest_docs.py:50-56`) — this is the complete configuration; nothing else is set:

```python
elements = partition_pdf(
    filename=str(pdf_file),
    strategy="hi_res",
    hi_res_model_name="yolox",   # comment in code: "Optimized for M3/CPU speed"
    chunking_strategy="by_title",
    max_characters=2000,
)
```

---

## 9. Chunking strategy

- **How chunks are formed:** by Unstructured's `by_title` strategy, inline during partitioning (§8). Elements are grouped into chunks that break at detected section titles.
- **Chunk size:** `max_characters=2000` — a hard cap; a chunk is split when it would exceed 2,000 characters.
- **Overlap:** **none configured.** No `overlap` or `overlap_all` parameter is passed, so adjacent chunks share no text. A sentence or table row split across a chunk boundary is retrievable from only one side.
- **Other by-title knobs not set:** `combine_text_under_n_chars` and `new_after_n_chars` are left at Unstructured's defaults. The exact defaults depend on the installed `unstructured` version (0.18.31 in the venv) — **Unknown / needs verification** without checking that library version's source.
- **Title-aware behavior:** chunks begin at titles, so a chunk usually holds one section's content — good for financial reports where numbers live under section headings.
- **Table handling:** weak. Without `infer_table_structure`, a table becomes flat text inside a chunk. Row/column relationships ("Revenue | 2024 | 2023") are likely garbled, which is a serious limitation for a financial system whose most important facts are in tables.
- **Page-number preservation:** yes — `metadata.get("page_number")` is stored per chunk (`ingest_docs.py:90`). For a chunk spanning pages, Unstructured reports one page number (typically the first).
- **Section-title preservation:** **not stored.** The chunk's text starts with the title text, but no `section` or `title` metadata field is written to the payload. You cannot filter or cite by section.
- **Duplicate-content risks:** the same fact often appears in multiple CIBC documents (news release, MD&A, annual report all state net income). All are ingested into one collection with no deduplication, so top-K results can be near-duplicates from different files, crowding out other relevant context.
- **Other limitations:** chunk cap is in characters not tokens; no minimum-chunk merging configured (tiny chunks possible); empty-text elements trigger the alignment bug described in §7.

---

## 10. Embedding models

Two models are actually wired into code. No others are referenced.

| | BAAI/bge-small-en-v1.5 | OpenAI text-embedding-3-small |
|---|---|---|
| Provider / runtime | Hugging Face model run **locally** via `fastembed` (ONNX, CPU/Apple Silicon) | **Hosted** OpenAI API |
| Vector dimension | 384 | 1536 |
| Named vector | `local_bge` | `openai` |
| Initialized (ingestion) | `TextEmbedding(...)` — `src/ingestion/ingest_docs.py:26` | `OpenAI(api_key=...)` inside `get_openai_embeddings()` — `ingest_docs.py:13-14` |
| Used during ingestion | `embedding_model.embed(texts)` — `ingest_docs.py:61` | `get_openai_embeddings(texts)` — `ingest_docs.py:63` |
| Initialized (query) | `FastEmbedWrapper.__init__` — `src/core/embeddings.py:16-17` | `OpenAI()` — `tests/test_retrieval.py:11` (manual script only) |
| Used during querying | **Yes — the only lane the agent uses.** `FastEmbedWrapper.embed_query()` returns `("local_bge", vector)` (`embeddings.py:19-27`), consumed by `LocalRetrievalTool` → `QdrantDatabase.search()` | **Not used by the agent.** Only exercised by the manual script `tests/test_retrieval.py:21-25`, which queries the stale `cibc_reports` collection |
| Cost | Free after one-time model download | Paid per token, per ingestion run (and the destructive re-ingest means paying again each run) |
| Privacy | Document text never leaves the machine | Full chunk text is sent to OpenAI during ingestion |
| Latency | Local CPU; no network | Network round trips, batched 100 texts per request |
| Current limitations | Small model; English-only; 512-token input limit typical for BGE-small (chunks of 2,000 chars ≈ 500 tokens are near that limit — tail truncation possible, **Unknown / needs verification**) | Requires `OPENAI_API_KEY` (not documented in `.env.example`); vectors are stored but never used for the agent's retrieval, so today the cost buys nothing at query time |

There is also a dead `EmbeddingModel` class in `src/core/models.py:144-154` that wraps the same BGE model; it is never imported (verified by grep).

**Why similarity scores from different embedding models cannot be directly compared:** each model maps text into its **own** vector space with its own geometry. A cosine similarity of 0.82 in BGE's 384-dimensional space and 0.70 in OpenAI's 1536-dimensional space are measured on different rulers — the numbers are not on a common scale, and different model families distribute scores differently (some cluster all scores between 0.6 and 0.9, others spread wider). The only fair comparison is **rank-based**: for the same query, does each model rank the truly relevant chunk at position 1, in the top 3, top 5, etc. (Hit@K, MRR, nDCG — see §17). Comparing raw scores across models says nothing.

---

## 11. Qdrant vector database

- **Where it runs:** Docker container `qdrant/qdrant:latest`, defined in `docker-compose.yml:5-13`. REST on port 6333, gRPC on 6334.
- **Local persistence:** compose mounts `./qdrant_storage:/qdrant/storage`, so data survives container restarts and lives in the repo folder (ignored by git).
- **Client connection:** `QdrantDatabase` (`src/core/database.py:23-52`) supports host:port, cloud URL+key, or embedded local path. In practice only host:port is used (`main.py:37`, `ingest_docs.py:24`). `settings.QDRANT_PATH` exists but is never used by any code.

**Collection names — every reference in the repo (verified by grep):**

| Name | Written by | Read by | Exists on disk? |
|---|---|---|---|
| `financial_reports` | `src/ingestion/ingest_docs.py:31,96` | `main.py:76` via `settings.QDRANT_COLLECTION` default (`config/settings.py:18`) | Yes — 21 MB |
| `cibc_reports` | nothing in current code (created by an earlier ingestion version) | `tests/test_retrieval.py:36` | Yes — 23 MB |
| `aequitas_documents` | nothing | nothing — appears **only** in `.env.example:5` as the suggested `QDRANT_COLLECTION` value | No |
| `documents` | nothing | default parameter of unused factory `create_tools()` (`src/reasoning/tools.py:180`) | No |

**Consistency verdict:** the *live* code path is internally consistent (ingestion writes and the agent reads `financial_reports`, because the actual `.env` does not set `QDRANT_COLLECTION` — verified by inspecting `.env` key names). But three inconsistencies remain: (1) the manual comparison script queries the stale `cibc_reports` collection, so its results describe old data; (2) anyone who copies `.env.example` to `.env` verbatim would point the agent at `aequitas_documents`, which does not exist — retrieval would silently return nothing (the tool logs a warning and returns `[]`, `tools.py:131-133`); (3) 23 MB of orphaned `cibc_reports` data sits on disk.

- **Named-vector configuration:** both on-disk collections have the identical schema — `local_bge` (384, Cosine) and `openai` (1536, Cosine) — verified from `qdrant_storage/collections/*/config.json`.
- **Payload structure** (written at `ingest_docs.py:83-92`): `{"text": <chunk text>, "metadata": {"source": <pdf filename>, "company": <str>, "fiscal_year": <str>, "element_type": <str>, "page_number": <int|null>}}`.
- **Metadata filters:** **none are ever used.** No code constructs a Qdrant `Filter`. Company/year metadata is write-only today.
- **Insertion behavior:** one `upsert` per PDF with fresh UUID point IDs — effectively insert-only.
- **Duplicate-ingestion behavior:** protected only by the destructive `recreate_collection` wipe. Remove the wipe without adding deterministic IDs and every re-run duplicates the corpus.
- **Query behavior:** `QdrantDatabase.search()` (`database.py:125-164`) calls `client.query_points(collection, query=vector, using=vector_name, limit=K)`. It accepts either a plain vector or a `(name, vector)` tuple for named-vector collections. Cosine similarity, no score threshold in practice (parameter exists, `main.py` doesn't set one), top-K=5 hardcoded at `main.py:78`.

---

## 12. Retrieval pipeline

Example: the user types *"What was CIBC's net income for fiscal 2024?"* into the `main.py` prompt.

1. **Query embedding.** `LocalRetrievalTool.retrieve()` (`src/reasoning/tools.py:119-157`) calls `FastEmbedWrapper.embed_query()` (`src/core/embeddings.py:19-27`), which runs the query through local BGE and returns `("local_bge", [384 floats])`.
2. **Named vector selection.** The tuple's first element tells `QdrantDatabase.search()` (`src/core/database.py:146-158`) to pass `using="local_bge"` to Qdrant — only the BGE vector space is searched. The stored OpenAI vectors are **never used** in this path.
3. **Top-K search.** Qdrant returns the 5 highest-cosine-similarity points from `financial_reports` (K hardcoded at `main.py:78`; the `RETRIEVAL_TOP_K` env setting exists but is not wired in).
4. **Score meaning.** Each hit's score is cosine similarity between the query vector and that chunk's `local_bge` vector — a relative closeness measure in BGE's space, roughly 0 to 1 for normalized embeddings. It is **not** a probability of correctness.
5. **Metadata returned.** Qdrant returns full payloads, but `retrieve()` keeps only `payload["text"]` (`tools.py:146-151`) and returns `List[str]`. Source file, page number, company, year, and score are all **discarded** before anyone downstream can use them.

**What does not exist (all planned at best, no code):**
- **Reranking** — none (no cross-encoder, no rerank step anywhere).
- **Hybrid search** — none.
- **BM25 / keyword search** — none.
- **Metadata filters** — none (no company or fiscal-year filter is applied, so the FY2024 question can return FY2025 chunks if both are in the collection).
- **Query rewriting** — a `QUERY_REFINEMENT_PROMPT` exists in `config/prompts.py:53-60` but is dead code.

The separate manual script `tests/test_retrieval.py` performs a different retrieval: it embeds one hardcoded query with **both** models and searches **both** named vectors (limit 3) against the stale `cibc_reports` collection, printing scores and excerpts for human comparison.

---

## 13. RAG implementation

The complete flow, as it actually exists:

**User question** → typed into the terminal loop at `main.py:99`, passed to `ReasoningGraph.run()` (`src/reasoning/graph.py:123-142`), which builds the initial `AgentState` via `create_initial_state()` (`src/reasoning/state.py:42-62`).

**→ Query processing** — there is none. No cleaning, no rewriting, no decomposition. The raw string flows through. The only "processing" is the router's keyword scan.

**→ Routing** — `router_node` (`src/reasoning/nodes.py:17-59`). A hand-written heuristic, **not** an LLM: if the query contains "current", "latest", "news", "today", "recent", or a year from now through four years ahead (computed dynamically from `datetime.now()`), it routes to web search; otherwise, if nothing has been retrieved yet, it routes to `retrieve`; otherwise `generate`. Failure mode: the keyword list is crude — *"What is CIBC's latest annual report?"* would be routed to web search even though the answer is local.

**→ Retrieval** — `rag_retrieval_node` (`nodes.py:62-87`) calls the retrieval described in §12 and puts `List[str]` into `state["retrieved_docs"]`. Failure modes: collection missing → empty list with only a log warning; any exception → empty list (`tools.py:155-157`); the user gets no signal that retrieval silently failed.

**→ Context construction** — inside `generate_answer_node` (`nodes.py:129-154`): each retrieved doc is truncated to 2,000 characters (`MAX_CONTEXT_LENGTH`, `nodes.py:14`) and numbered under a "## Local Knowledge Base:" heading; up to 3 web results (also truncated) go under "## Web Search Results:". No sources, pages, or scores are included — they were already discarded. Failure mode: with 5 docs × 2,000 chars plus prompt text, context can approach small local models' limits; nothing counts tokens.

**→ Prompt creation** — an **inline f-string** at `nodes.py:157-164`: *"Based on the following context, please answer the user's query. … If the context doesn't contain enough information, acknowledge this."* Note: the polished templates in `config/prompts.py` (including a system prompt and a citation instruction) are **not used**. There is no system message at all.

**→ LLM generation** — `llm.invoke([HumanMessage(content=prompt)])` (`nodes.py:168`), where `llm` is a `ChatOllama` from `LLMModels.get_default_model()` (`src/core/models.py:99-115`), temperature 0.7, `num_predict` 2048. Failure modes: Ollama not running or model not pulled → exception is caught and the user gets a generic apology string (`nodes.py:171-173`); temperature 0.7 invites paraphrased or hallucinated numbers; nothing verifies the answer against the context.

**→ Final answer** — stored in `state["answer"]`, printed by `main.py:113-116` along with counts of retrieved docs/web results and the iteration count. No citations, no confidence, no grounding check.

**Loop control:** after `retrieve` or `search_web`, the graph returns to the router; `route_next_node` (`nodes.py:185-210`) forces `generate` once `iteration >= max_iterations` (default 5, from `settings.MAX_ITERATIONS`). `generate` always ends the graph (`graph.py:118`).

---

## 14. LLM and model configuration

- **Ollama usage:** the only chat-model backend. `ChatOllama` (from `langchain-ollama`) is constructed in `LLMModels` (`src/core/models.py:24-115`) with `base_url=settings.OLLAMA_BASE_URL` (default `http://localhost:11434`). Ollama itself must be installed and running separately — it is **not** in `docker-compose.yml`.
- **Default model:** `settings.DEFAULT_MODEL` (`config/settings.py:22`) — env var `DEFAULT_MODEL`, code default `"llama3"`. `.env.example:9` suggests `llama3.1:8b`. The actual local `.env` does not set `DEFAULT_MODEL` (verified by key inspection), so the running default is `"llama3"`. Commit `6ac36e5` ("update default Ollama model") changed the example, not the code default.
- **CLI override:** `python main.py --model <tag>` (`main.py:22-26`) takes precedence over the env var. Added in commit `65093e5`.
- **Temperature:** `MODEL_TEMPERATURE`, default **0.7** (`settings.py:23`). High for factual financial Q&A; nothing overrides it per-request.
- **Max tokens:** `MODEL_MAX_TOKENS`, default 2048, passed to `ChatOllama` as `num_predict` (`models.py:113-114`).
- **Prompts:** one inline user-message prompt in `nodes.py:157-164`. No system prompt is sent. `config/prompts.py` is dead code (see §13).
- **Model provider abstraction:** thin. `LLMModels` has hardcoded `get_llama3()` / `get_mistral()` helpers (both unused) and the generic `get_default_model(model_name)` (used). Everything assumes Ollama; there is no interface for swapping in another chat provider.
- **Is OpenAI or Anthropic generation supported?** **No.** The `openai` package is used **only for embeddings** (`ingest_docs.py`, `tests/test_retrieval.py`) — never for chat/completion. There is no Anthropic code at all. All answer generation is local Ollama. (This is the embedding-vs-chat distinction: OpenAI is in the ingestion lane, not the generation lane.)

---

## 15. LangGraph or reasoning workflow

- **Graph state:** `AgentState` TypedDict (`src/reasoning/state.py:9-39`): `query`, `messages` (annotated with `operator.add` so lists concatenate), `retrieved_docs`, `web_results`, `next_action`, `answer`, `iteration`, `max_iterations`.
- **Nodes** (added in `ReasoningGraph._build_graph`, `src/reasoning/graph.py:57-121`): `router`, `retrieve` (only if a retrieval tool was passed), `search_web` (only if Tavily is configured), `generate`.
- **Edges:** entry point → `router`; conditional edges from `router` via `route_next_node` to `retrieve` / `search_web` / `generate` (unavailable tools fall back to `generate`, `graph.py:100-108`); `retrieve` → `router`; `search_web` → `router`; `generate` → `END`.
- **Tools:** `LocalRetrievalTool` and `TavilySearchTool` (`src/reasoning/tools.py`). Note: they are invoked directly inside node functions — the LLM never sees or chooses tools. `as_langchain_tool()` conversions exist but are unused.
- **Routing:** keyword heuristic in Python (`nodes.py:33-51`), not an LLM decision. The code comment itself says "In production, this could use an LLM."
- **Fallbacks:** missing tools reroute to `generate`; retrieval/search exceptions return empty lists; generation exceptions return an apology string. No retry logic anywhere.
- **Termination:** guaranteed — `route_next_node` forces `generate` at `iteration >= max_iterations`, and `generate` always reaches `END`.
- **Autonomous agent or deterministic workflow?** **Deterministic workflow.** Given the same query and state, the path through the graph is fully determined by if/else rules. The LLM makes no decisions about what to do — it only writes the final answer. Calling this an "agent" in an interview would be overstating it; it is a routed RAG pipeline built on LangGraph's state machine.

```
                 ┌──────────────┐
   user query ──▶│    router    │  (keyword heuristic, not LLM)
                 └──────┬───────┘
             ┌──────────┼──────────────┐
             ▼          ▼              ▼
      ┌──────────┐ ┌────────────┐ ┌──────────┐
      │ retrieve │ │ search_web │ │ generate │──▶ END
      │ (Qdrant, │ │  (Tavily,  │ │ (Ollama) │
      │local_bge)│ │  optional) │ └──────────┘
      └────┬─────┘ └─────┬──────┘      ▲
           │             │             │
           └──────▶ router ◀───────────┘
              (loops until docs gathered
               or max_iterations reached)
```

---

## 16. Financial tools and deterministic calculations

**There are none.** Verified by inspecting every module: the repo contains no ratio calculations, no table math, no numeric extraction, no currency handling — nothing that computes a financial quantity deterministically.

| Tool | What it does | Input | Output | Used? | Tests? | Limitations |
|---|---|---|---|---|---|---|
| `LocalRetrievalTool` (`src/reasoning/tools.py:89`) | Semantic search over Qdrant (not a financial calculator) | query string | list of chunk texts | Yes (`main.py:74`) | No | Discards metadata/scores |
| `TavilySearchTool` (`tools.py:12`) | Web search via Tavily API (not a financial calculator) | query string | list of formatted snippets | Only if `TAVILY_API_KEY` set | No | Requires paid API key |

Any numerical claim in an answer today comes straight from LLM text generation over retrieved chunks — with temperature 0.7 and no verification. This is the single biggest gap between the project's name/ambition ("financial intelligence") and its implementation. Deterministic calculation tools (e.g., "compute YoY change from two retrieved figures") are **Planned** at best; nothing exists.

---

## 17. Evaluation

**What `src/evaluation` contains:** two files, `metrics.py` and `run_evals.py`, both **0 bytes** (verified). The package is an empty placeholder and is never imported.

**Is it used?** No.

**The manual retrieval comparison script:** `tests/test_retrieval.py` defines `run_comparison_test(query_text)` and runs it under `if __name__ == "__main__":` with the single hardcoded query *"What was CIBC's net income for the full year 2025?"*. It embeds the query with both models, searches both named vectors on the `cibc_reports` collection (limit 3), and **prints** scores, element types, sources, page numbers, and excerpts. A human reads the output and judges quality. There are no assertions and no ground truth.

**Why pytest collects zero tests** (verified: `pytest --collect-only` → "no tests collected"):
1. The only function is named `run_comparison_test` — pytest only collects functions matching `test_*`.
2. There are no assert statements, so even if collected it would test nothing.
3. There is no `pytest.ini`/`pyproject.toml` configuration and no conftest.
4. The script needs a live Qdrant on localhost **and** a valid `OPENAI_API_KEY` — it would fail in any clean environment anyway.

**What the 0.816 score actually means.** The README's performance table (`README.md:21-28`) reports "Search Score 0.8160" for the local BGE lane. This is the **cosine similarity between one query's BGE embedding and the single top-ranked chunk's BGE embedding, for one query, on one corpus**. It says the top chunk was geometrically close to the query in BGE's vector space — and the human running the script judged that chunk relevant (Q4 news release, page 1).

**Why it is not accuracy:**
- It is one query. Accuracy is a rate over many queries.
- It measures vector closeness, not answer correctness. A wrong-but-similar chunk also scores high.
- There is no labeled ground truth — nobody defined which chunk *should* rank first.
- The number is model-specific (see §10) — the paired OpenAI score of 0.0000 in the same table reflects that OpenAI vectors were zero placeholders at the time, a "Dummy State" as the README itself says. Real OpenAI embeddings landed later (commit `cb4300b`); the table was never re-run, and the README roadmap still lists the placeholder replacement as an open item. **The README's evaluation section is stale.**

**Correct future benchmark metrics** (what a real evaluation should report, none of which exist today):

| Metric | Meaning |
|---|---|
| Hit@1 | Fraction of queries where the correct chunk is ranked first |
| Hit@3 / Hit@5 | Fraction where the correct chunk appears in the top 3 / top 5 |
| Recall@K | Fraction of all relevant chunks that appear in the top K |
| MRR | Mean Reciprocal Rank — average of 1/(rank of first correct chunk); rewards ranking the right chunk high |
| nDCG@K | Rank-quality score that gives more credit for relevant chunks nearer the top, with graded relevance |
| Latency | Time per query (embed + search, and separately end-to-end with generation) |
| Cost | API dollars per ingestion run and per query, tokens consumed |
| Answer correctness | Does the generated answer state the right figure? (judged against a gold answer set) |
| Citation correctness | Do cited sources/pages actually contain the claimed fact? |
| Unsupported-claim rate | Fraction of answer sentences not backed by any retrieved chunk (hallucination rate) |

Building this requires a labeled query→chunk/answer gold set — a natural fit for the currently empty `data/gold/` folder.

---

## 18. Current automated test coverage

- **Discovered tests:** **zero.** Verified by running `venv/bin/python -m pytest --collect-only -q` → `no tests collected`.
- **Undiscovered scripts:** one — `tests/test_retrieval.py`. Despite the filename, it is a manual comparison script (see §17). Its presence in `tests/` should not be described as test coverage.
- **Assertions:** zero in the entire repository.
- **External-service dependencies of the one script:** live Qdrant at `localhost:6333`, OpenAI API (`OPENAI_API_KEY`), first-run download of the BGE model. None are mocked.
- **Missing unit tests (highest-value candidates):** metadata derivation from folder paths (`ingest_docs.py:65-69` — including the wrong-depth failure case); the element/embedding zip alignment (`ingest_docs.py:59-73` — would catch the confirmed bug); `router_node` keyword routing; `route_next_node` iteration cap; `QdrantDatabase.search` tuple/plain-vector handling; context truncation in `generate_answer_node`.
- **Missing integration tests:** ingest a small fixture PDF into a throwaway Qdrant (or embedded `QdrantClient(path=...)`, which `QdrantDatabase` already supports) and assert point counts, payload fields, and both named vectors present; retrieval round-trip returning the planted chunk.
- **Missing end-to-end tests:** scripted query through `ReasoningGraph.run()` with a stubbed LLM, asserting an answer is produced and retrieved docs flow into the prompt.
- Also missing: any CI configuration (no GitHub Actions or other CI files exist).

---

## 19. Configuration and dependencies

**Every environment variable, where it is read, and its status:**

| Variable | Read at | Default | Required? | Actually used? | In `.env.example`? | In local `.env`? (keys verified, values not inspected) |
|---|---|---|---|---|---|---|
| `QDRANT_HOST` | `config/settings.py:15` | `localhost` | No | Yes (`main.py:37`) — but `ingest_docs.py:24` ignores it and hardcodes localhost | Yes | Yes |
| `QDRANT_PORT` | `settings.py:16` | `6333` | No | Yes (`main.py:37`) — ditto hardcoded in ingestion | Yes | Yes |
| `QDRANT_PATH` | `settings.py:17` | `./qdrant_storage` | No | **Never used by any code** | Yes | No |
| `QDRANT_COLLECTION` | `settings.py:18` | `financial_reports` | No | Yes (`main.py:76`) | Yes — **but with the wrong value `aequitas_documents`**, which would break retrieval | No (so the safe default applies) |
| `OLLAMA_BASE_URL` | `settings.py:21` | `http://localhost:11434` | No | Yes (`main.py:43`) | Yes | No |
| `DEFAULT_MODEL` | `settings.py:22` | `llama3` | No | Yes (`main.py:26`) | Yes (`llama3.1:8b`) | No |
| `MODEL_TEMPERATURE` | `settings.py:23` | `0.7` | No | Yes (`main.py:44`) | Yes | No |
| `MODEL_MAX_TOKENS` | `settings.py:24` | `2048` | No | Yes (`main.py:45`) | Yes | No |
| `TAVILY_API_KEY` | `settings.py:27` | `""` | Optional (web search disabled without it) | Yes (`main.py:55-59`) | Yes | Yes |
| `TAVILY_MAX_RESULTS` | `settings.py:28` | `5` | No | Yes (`main.py:59`) | Yes | No |
| `RETRIEVAL_TOP_K` | `settings.py:31` | `5` | No | **Never used** — `main.py:78` hardcodes `top_k=5` | Yes | No |
| `VECTOR_SIZE` | `settings.py:32` | `384` | No | **Never used** | Yes | No |
| `MAX_ITERATIONS` | `settings.py:35` | `5` | No | Yes (`main.py:110`) | Yes | No |
| `OPENAI_API_KEY` | `ingest_docs.py:14` (`os.getenv`), implicitly by `OpenAI()` in `tests/test_retrieval.py:11` | none | **Required for ingestion and the comparison script** | Yes | **No — missing from `.env.example`** | Yes |

Summary of config problems: three dead settings (`QDRANT_PATH`, `RETRIEVAL_TOP_K`, `VECTOR_SIZE`); one required variable undocumented (`OPENAI_API_KEY`); one documented variable with a value that breaks the app (`QDRANT_COLLECTION=aequitas_documents`); ingestion bypasses settings entirely for host/port.

**Imports vs. `requirements.txt`:**

Declared (`requirements.txt`): `langchain`, `langgraph`, `langchain-community`, `qdrant-client`, `tavily-python`, `langchain-ollama`, `python-dotenv`, `pydantic`.

| Package | Imported in code? | Declared? | Verdict |
|---|---|---|---|
| `unstructured` (+ its `hi_res` extras: `unstructured-inference`, YOLOX weights, and system deps like poppler/tesseract) | Yes — `ingest_docs.py:5` | **No** | **Missing dependency.** A fresh `pip install -r requirements.txt` cannot run ingestion. (Installed in the local venv: `unstructured 0.18.31`.) |
| `fastembed` | Yes — `embeddings.py:4`, `models.py:8`, `ingest_docs.py:6`, `test_retrieval.py:4` | **No** | **Missing dependency** (local venv has 0.7.4). |
| `openai` | Yes — `ingest_docs.py:7`, `test_retrieval.py:5` | **No** | **Missing dependency** (local venv has 2.16.0). |
| `pytest` | Needed to run tests | No (no dev requirements file) | Missing dev dependency. |
| `langchain-community` | **Never imported** | Yes | Declared but unused directly. |
| `pydantic` | **Never imported directly** (only transitively via langchain/qdrant) | Yes | Unnecessary as a direct pin. |

**Version-range risks:** every declared range is broad (e.g. `langchain>=0.1.0,<1.0.0` — the venv resolved to 0.3.27; `langgraph>=0.0.20,<1.0.0` → 0.6.11). These majors-wide ranges mean a fresh install months from now can pull significantly different APIs than the code was written against. `from langchain.schema import ...` (`nodes.py:7`, `state.py:5`, `models.py:6`) is a deprecated import path that still works in langchain 0.3 but is a break risk. There is no lock file.

---

## 20. Docker and local services

`docker-compose.yml` defines exactly two services:

| Service | Image | Ports | Volume | Integrated with the code? |
|---|---|---|---|---|
| `qdrant` | `qdrant/qdrant:latest` | 6333 (REST), 6334 (gRPC) | `./qdrant_storage:/qdrant/storage` | **Yes** — ingestion and the agent both connect to it on 6333. gRPC 6334 is exposed but unused by the code. |
| `phoenix` | `arizephoenix/phoenix:latest` | 6006 (UI/OTLP HTTP), 4317 (OTLP gRPC) | none (traces would not survive a restart) | **No** — merely present. Zero references to Phoenix, Arize, or OpenTelemetry anywhere in Python code (verified by grep). |

- **Agent container:** does not exist. `.docker/Dockerfile.agent` is a tracked but **completely empty** file, and no compose service references it. The application runs only on the host Python.
- **Ollama:** required by the app but **not** in compose — it must be installed and running on the host separately.
- **Obsolete configuration:** the top-level `version: '3.8'` key is deprecated in current Docker Compose (harmless warning); `image: ...:latest` on both services makes environments unreproducible; the empty `Dockerfile.agent` is dead weight.
- **Startup commands:** `docker-compose up -d` (per `README.md:64`), then separately: Ollama (`ollama serve` + model pulled), `PYTHONPATH=. python src/ingestion/ingest_docs.py`, `python main.py`.

---

## 21. Observability

| Layer | Status |
|---|---|
| Phoenix (tracing UI) | **Configured** in docker-compose, **referenced nowhere in code, unused**. Starting the container gives an empty dashboard; the app never sends a span to it. |
| OpenTelemetry / tracing instrumentation | **Absent** — no packages, no imports. |
| Structured logging | **Absent.** `logging.basicConfig` at `main.py:14-17` produces plain-text INFO logs to the console only. Modules use `logger.info/debug/warning/error` consistently (a good base), but there is no file output, no JSON, no request IDs. `src/ingestion/ingest_docs.py` and `tests/test_retrieval.py` use bare `print()`. |
| Metrics | **Absent.** |
| Cost tracking | **Absent** — OpenAI embedding spend and token usage are not recorded anywhere. |

Plainly: nothing is actually wired in. The only observability today is reading console output.

---

## 22. Security and privacy

- **API keys:** `OPENAI_API_KEY` and `TAVILY_API_KEY` live in the local `.env`. Verified that `.env` is in `.gitignore` and **not tracked** (`git ls-files` shows only `.env.example`). No secrets appear in any committed file, and no key values were found hardcoded in code. **Caveat:** git history was checked only at the file-list level; a full secret scan of all historical blobs was not performed — **Unknown / needs verification** (running a secret scanner over history is cheap insurance).
- **`.env` handling:** loaded via `python-dotenv` (`settings.py:8`, `ingest_docs.py:11`, `test_retrieval.py:7`). `.env.example` is incomplete (missing `OPENAI_API_KEY`) and partly wrong (`QDRANT_COLLECTION` value) — see §19.
- **Financial-document privacy:** the current corpus (CIBC filings) is **public** data, so exposure risk today is low. But the architecture sends full chunk text to OpenAI during ingestion (`ingest_docs.py:18`) — if a user drops confidential documents into `data/raw/`, that content leaves the machine. The "local lane" (BGE + Ollama) is the private path; the OpenAI lane is not.
- **External calls inventory:** OpenAI (ingestion embeddings + comparison script), Tavily (optional web search — the raw user query is sent to Tavily, `tools.py:48-51`), Hugging Face (one-time BGE model download), Ollama (localhost only).
- **Local model use:** query embedding and answer generation are fully local — good default posture.
- **Prompt injection risk:** real and unmitigated. PDF-extracted text and Tavily web snippets are pasted directly into the LLM prompt (`nodes.py:136-164`). A malicious document or web page could contain instructions the model follows. Impact is bounded today (the LLM controls no tools and outputs only text), but any future tool-use upgrade makes this serious.
- **Data leakage via logs:** low risk currently — logs record query prefixes (first 50 chars) and counts, not full document contents. `print()` in the comparison script shows 150-char excerpts to the local console only.
- **Unsafe logging:** none observed; no secrets are logged.
- **Other:** no input validation on the CLI query (low risk — it only becomes an embedding and prompt text); Qdrant and Phoenix listen on localhost Docker ports with no authentication (fine locally, unacceptable if ever exposed).

---

## 23. Current limitations and technical debt

### Critical (prevent trustworthy use)

1. **Destructive, non-idempotent ingestion.**
   Evidence: `client.recreate_collection(...)` at `src/ingestion/ingest_docs.py:30`; random UUID point IDs at `:78`. Affected: `src/ingestion/ingest_docs.py`.
   Impact: every ingestion run erases all stored data (and the OpenAI embedding spend that produced it); adding a second company or year requires re-ingesting everything; a crash mid-run (no try/except around `partition_pdf`) leaves a half-empty collection.
   Next action: derive deterministic point IDs (e.g., UUID5 of file hash + chunk index), create the collection only if missing, and never wipe implicitly.

2. **Element/embedding misalignment bug.**
   Evidence: `texts` filters out empty-text elements (`ingest_docs.py:59`) but the point loop zips the **unfiltered** `elements` with the embeddings (`:73`).
   Impact: if any element has blank text, every subsequent chunk is stored with the wrong vectors — retrieval silently returns wrong-ranked chunks. Whether the current corpus triggered it: Unknown / needs verification.
   Next action: build one filtered list of elements first, embed from it, zip over it; add a unit test.

3. **Collection-name inconsistency (three names).**
   Evidence: ingestion/agent use `financial_reports` (`ingest_docs.py:31`, `settings.py:18`); comparison script uses `cibc_reports` (`test_retrieval.py:36`); `.env.example:5` says `aequitas_documents`. Both `cibc_reports` and `financial_reports` exist on disk.
   Impact: the published comparison results describe a stale collection; copying `.env.example` breaks retrieval silently (empty results, warning only in logs).
   Next action: standardize on one name from `settings.QDRANT_COLLECTION` everywhere; fix `.env.example`; delete or archive `cibc_reports`.

4. **No evaluation and no tests at all.**
   Evidence: `src/evaluation/metrics.py` and `run_evals.py` are 0 bytes; pytest collects zero tests (verified).
   Impact: no way to know if any change improves or degrades retrieval or answers; the 0.816 headline number is a single-query similarity score, not a metric.
   Next action: milestones 6–7 in §25.

5. **`requirements.txt` cannot reproduce the environment.**
   Evidence: `unstructured`, `fastembed`, `openai` imported (`ingest_docs.py:5-7`, `embeddings.py:4`) but not declared.
   Impact: a fresh clone + install fails at ingestion with `ModuleNotFoundError`; interviews/demos on another machine break.
   Next action: regenerate requirements (with the `unstructured[pdf]` extra), add a dev-requirements file with pytest, consider a lock file.

6. **Ungrounded answers with no citations at temperature 0.7.**
   Evidence: metadata discarded at `tools.py:146-151`; inline prompt without citation instructions at `nodes.py:157-164`; `MODEL_TEMPERATURE=0.7` default (`settings.py:23`).
   Impact: for a financial Q&A tool, unverifiable numbers are the core trust failure.
   Next action: return payload dicts (text + source + page + score) from retrieval, cite in the prompt, lower temperature to ~0–0.2.

### Important (reduce quality, maintainability, reproducibility)

7. **Dead code and dead config mislead readers.** `config/prompts.py` (never imported), `EmbeddingModel`/`get_llama3`/`get_mistral`/`get_llm` (`models.py`), `create_tools` (`tools.py:177-214`), `QdrantDatabase.create_collection`/`upsert_vectors` (bypassed by ingestion), settings `QDRANT_PATH`/`RETRIEVAL_TOP_K`/`VECTOR_SIZE`, empty `.docker/Dockerfile.agent`, `models.py` docstring claiming "placeholder implementations". Next action: delete or wire in, one by one.
8. **README is stale/contradictory.** Roadmap still lists the OpenAI-placeholder fix that shipped in `cb4300b`; performance table reflects the placeholder era (OpenAI 0.0000); ingestion instructions point at `FY2025` while the corpus is `FY2024`; the reasoning layer now exists but is listed as future work. Next action: rewrite README from this document.
9. **Ingestion ignores `Settings`.** Hardcoded `localhost:6333`, hardcoded `data/raw/cibc`, hardcoded collection name (`ingest_docs.py:24,31,100`). Next action: parameterize via `config.settings` + CLI args.
10. **No metadata filtering at query time.** Company/year stored but never filtered on; multi-year corpora will cross-contaminate answers. Next action: add Qdrant `Filter` support to `QdrantDatabase.search` and `LocalRetrievalTool`.
11. **Table structure is lost.** No `infer_table_structure`; financial tables become flat text. Next action: enable table inference and store table HTML in payloads.
12. **`.gitignore` gaps + stray `EOF` line;** `data/silver/`, `data/raw_backup/` untracked-but-not-ignored. Next action: clean up ignore rules.
13. **OpenAI vectors cost money but are unused at query time.** Only `local_bge` is ever searched by the agent. Next action: either use the lane (benchmark it properly) or make it optional at ingest.
14. **Keyword router misroutes.** "latest annual report" → web search; hardcoded English keywords. Acceptable for a prototype; document it.

### Optional (can wait)

15. Phoenix container unused — remove it or wire OpenTelemetry in (see §25 milestone 13).
16. `version: '3.8'` and `:latest` tags in docker-compose; pin versions.
17. Deprecated `from langchain.schema import ...` paths and deprecated `recreate_collection` API.
18. Filename typo `inventorPresentation2024.pdf`; `notebooks/` and `mlx_models/` empty dirs.
19. `main.py` hardcodes `top_k=5` instead of `settings.RETRIEVAL_TOP_K`.
20. gRPC port 6334 exposed but unused.

---

## 24. Target architecture

This is the **recommended** end state. None of it should be described as existing today unless cross-referenced above.

- **Ingestion:** a CLI (`ingest --company cibc --year 2024`) that walks `data/raw/{company}/FY{year}/`, partitions with hi_res + `infer_table_structure=True`, chunks by title with overlap, writes chunks with rich metadata (company, fiscal year, source file, page, section title, element type, ingest timestamp, content hash), uses deterministic IDs, and is safely re-runnable (skip unchanged files by hash).
- **Storage:** one Qdrant collection with named vectors per embedding model under evaluation, payload indexes on `company` and `fiscal_year` for fast filtering.
- **Retrieval:** metadata-filtered hybrid search (dense + BM25/sparse), followed by a cross-encoder reranker; retrieval returns structured hits (text, score, source, page).
- **Reasoning:** LangGraph flow with an LLM-based router, query rewriting on weak retrieval, deterministic financial calculation tools (ratio math, YoY deltas) the graph can call, and a grounding/citation-validation step that checks every numeric claim appears in a cited chunk.
- **Generation:** low-temperature local LLM by default (privacy), optional hosted model behind a provider abstraction, prompts sourced from `config/prompts.py` (finally used).
- **Evaluation:** a committed gold set in `data/gold/` (queries → relevant chunk IDs → gold answers); `run_evals.py` computes Hit@K, Recall@K, MRR, nDCG, latency, and cost per embedding model; answer-level scoring for correctness, citation correctness, and unsupported-claim rate; results written to versioned reports.
- **API/UI:** a FastAPI service exposing `/query` and `/ingest`, and a thin UI (or notebook) on top.
- **Observability:** OpenTelemetry traces to Phoenix (finally justifying the container), structured JSON logs, token/cost counters per request.
- **Deployment:** the app containerized (a real `Dockerfile.agent`), compose running qdrant + phoenix + api with pinned image versions; CI running lint + tests + a smoke eval on every PR.

```
                        ┌────────────────────────────────────────────┐
                        │              docker compose                 │
 data/raw/{co}/FY{yr}/  │  ┌─────────┐   ┌─────────┐   ┌──────────┐  │
 ─── PDFs ──▶ ingest ───┼─▶│ Qdrant  │◀──│ FastAPI  │──▶│ Phoenix  │  │
     CLI (unstructured  │  │ (dense+ │   │  /query  │   │ (traces) │  │
     hi_res + tables,   │  │ sparse, │   └────┬─────┘   └──────────┘  │
     deterministic IDs) │  │ filters)│        │                       │
                        │  └─────────┘        ▼                       │
                        └───────────── LangGraph ─────────────────────┘
                                       │  router (LLM) → rewrite
                                       │  hybrid retrieve → rerank
                                       │  fin-calc tools (deterministic)
                                       │  generate (low-temp, cited)
                                       │  grounding check
                                       ▼
                          answer + citations + confidence
                                       ▲
                          eval harness (gold set in data/gold/):
                          Hit@K · Recall@K · MRR · nDCG · latency ·
                          cost · answer/citation correctness
```

---

## 25. Recommended development roadmap

Ordered, small milestones. Each is a PR-sized unit of work (per the required git workflow: branch off `main`, merge via PR).

1. **Stabilize repository and dependencies.**
   Goal: a fresh clone runs. Files: `requirements.txt` (add `unstructured[pdf]`, `fastembed`, `openai`; add `requirements-dev.txt` with pytest), `.env.example` (add `OPENAI_API_KEY`, fix `QDRANT_COLLECTION`), `.gitignore` (remove `EOF`, cover `data/`), delete `.docker/Dockerfile.agent` or fill it, delete dead code (`config/prompts.py` or wire it in, unused classes in `models.py`/`tools.py`).
   Success: `pip install -r requirements.txt` then ingestion imports cleanly on a clean machine.
   Learn: what the true dependency surface is (especially unstructured's system deps).

2. **Standardize raw-data layout.**
   Goal: `data/raw/{company}/FY{year}/*.pdf` is documented and enforced. Files: `README.md`, small validator in `src/ingestion/`.
   Success: ingestion refuses (with a clear error) PDFs at the wrong depth instead of writing wrong metadata.
   Learn: how strict path conventions prevent silent metadata corruption.

3. **Fix ingestion metadata and the alignment bug.**
   Goal: every stored point has correct text↔vector pairing and full metadata (add section title, ingest timestamp, content hash). Files: `src/ingestion/ingest_docs.py`.
   Success: unit test with a fixture containing an empty element passes; spot-check of stored payloads.
   Learn: how chunk metadata quality drives everything downstream.

4. **Make ingestion idempotent.**
   Goal: re-running ingestion never wipes or duplicates. Files: `src/ingestion/ingest_docs.py`, `src/core/database.py`.
   Success: run ingestion twice → identical point count; adding one new PDF adds only its chunks.
   Learn: deterministic IDs (content-hash based) vs. UUIDs.

5. **Resolve Qdrant collection design.**
   Goal: one canonical collection name, from settings, everywhere; payload indexes for `company`/`fiscal_year`; drop `cibc_reports`. Files: `ingest_docs.py`, `tests/test_retrieval.py`, `settings.py`, `.env.example`.
   Success: grep finds exactly one collection-name source of truth.
   Learn: config-driven design vs. hardcoding.

6. **Create real automated tests.**
   Goal: pytest collects and passes a suite with no external services (embedded Qdrant via `QdrantClient(path=...)`, stub LLM). Files: `tests/` (new unit + integration tests), possibly `pyproject.toml`.
   Success: `pytest` green in a clean venv; router, metadata, alignment, and search covered.
   Learn: testing seams — where the code needs interfaces to be testable.

7. **Build the embedding benchmark.**
   Goal: replace the 0.816 anecdote with a real comparison. Files: `data/gold/queries.jsonl` (20–50 labeled queries), `src/evaluation/metrics.py` (Hit@K, Recall@K, MRR, nDCG), `src/evaluation/run_evals.py` (runs both named vectors, reports table + latency + cost).
   Success: one command prints a per-model metrics table; results committed to `docs/`.
   Learn: whether the 4× cheaper local lane actually loses to OpenAI — with evidence.

8. **Improve retrieval.**
   Goal: metadata filters (company/year), then hybrid (BM25/sparse) and a reranker, each measured against the milestone-7 benchmark. Files: `src/core/database.py`, `src/reasoning/tools.py`.
   Success: benchmark metrics improve and the deltas are recorded.
   Learn: which retrieval upgrades pay off on financial PDFs.

9. **Add grounded answer generation.**
   Goal: retrieval returns structured hits; prompt includes sources and demands citations; temperature ≈ 0.1. Files: `tools.py`, `nodes.py`, `config/prompts.py` (now actually used).
   Success: every answer lists source file + page for each claim.
   Learn: prompt design for grounding local models.

10. **Add answer evaluation.**
    Goal: gold answers for benchmark queries; score answer correctness, citation correctness, unsupported-claim rate. Files: `src/evaluation/`, `data/gold/`.
    Success: an end-to-end quality score exists and is tracked over time.
    Learn: how retrieval metrics do (and don't) predict answer quality.

11. **Add API.**
    Goal: FastAPI `/query` endpoint wrapping `ReasoningGraph`. Files: new `src/api/`.
    Success: `curl` returns answer + citations JSON.
    Learn: separating the engine from the interface.

12. **Add interface.**
    Goal: minimal UI (even Streamlit) over the API.
    Success: a non-developer can ask a question.
    Learn: what users actually ask, feeding the gold set.

13. **Add observability.**
    Goal: OpenTelemetry tracing into the already-provisioned Phoenix; structured logs; token/cost counters. Files: `main.py`/`src/api/`, `docker-compose.yml` (Phoenix volume).
    Success: a query's full trace (route → retrieve → generate, with timings) is visible in Phoenix.
    Learn: where latency actually goes.

14. **Containerize and document deployment.**
    Goal: real `Dockerfile.agent`, compose service for the API, pinned versions, CI running tests + smoke eval.
    Success: `docker compose up` yields a working system; a fresh machine needs only Docker + an Ollama model.
    Learn: reproducibility end to end.

---

## 26. Interview-ready explanation

*A concise, honest narrative you can say out loud:*

"Aequitas-Fin is a RAG system I'm building for question-answering over financial filings — the initial corpus is CIBC's annual-report package. The business problem: the facts analysts need are buried in hundreds of pages of layout-heavy PDFs, and most of the important numbers live in tables that naive text extraction destroys.

Architecturally it's a local-first pipeline: Unstructured.io parses each PDF with its `hi_res` strategy — a layout-detection model rather than plain text extraction — because financial reports are visually structured documents; I chunk by section title so each chunk stays topically coherent. Chunks go into Qdrant, which I chose because its **named vectors** let me store two embeddings per chunk in one collection: a local 384-dim BGE-small vector computed on-device, and a 1536-dim OpenAI `text-embedding-3-small` vector. That gives me an apples-to-apples A/B setup on identical chunks — the real question being whether a free, private, local embedding model is good enough, or whether the hosted model's extra dimensions buy real retrieval quality. On the generation side, a LangGraph workflow routes between local retrieval, optional web search, and answer generation with a local Ollama model, so document content never has to leave the machine at query time.

About the 0.816 number: that's the cosine similarity of the top-ranked chunk for **one** test query on the local BGE lane — it told me the pipeline retrieves sensibly, but it's a single-query similarity score, not an accuracy metric, and at the time the OpenAI lane still held placeholder vectors, so no real comparison has happened yet. I'm careful not to oversell it.

Current honest limitations: ingestion wipes and rebuilds the collection each run, so it isn't idempotent; there's no labeled evaluation set, so no Hit@K/MRR numbers yet; answers aren't cited because retrieval currently drops the page/source metadata before generation; tables are ingested as flat text; and pytest collects zero tests — the one 'test' file is a manual comparison script. The routing is also a keyword heuristic, so it's a deterministic workflow, not an autonomous agent.

Next steps, in order: make ingestion idempotent with deterministic IDs, build a 30–50 query gold set and score both embedding models on Hit@K, MRR, and nDCG, add metadata-filtered and hybrid retrieval with reranking, and wire citations plus a grounding check into generation so every number in an answer traces to a page in a filing."

---

## 27. Glossary

- **RAG (Retrieval-Augmented Generation):** instead of asking an LLM to answer from memory, you first *retrieve* relevant document passages and paste them into the prompt, so the model answers *from your documents*.
- **Embedding:** a list of numbers (a vector) that represents a text's meaning. Texts with similar meaning get vectors that point in similar directions.
- **Vector database:** a database (here, Qdrant) built to store millions of embeddings and quickly find the ones closest to a query vector.
- **Cosine similarity:** a score for how similar two vectors' directions are — 1 means identical direction, 0 means unrelated. It measures closeness in embedding space, not factual correctness.
- **Named vectors:** a Qdrant feature where one stored record holds several vectors under different names (here `local_bge` and `openai`), so different embedding models can be compared on identical data.
- **Chunking:** splitting long documents into smaller pieces so each piece can be embedded and retrieved on its own.
- **Title-aware chunking:** starting a new chunk at each section heading, so a chunk holds one coherent topic instead of an arbitrary text window.
- **Reranking:** a second, more careful scoring pass (usually a cross-encoder model) that reorders the retriever's top results. Not implemented here.
- **Grounding:** making sure every claim in a generated answer is supported by a retrieved source. Not implemented here.
- **Hallucination:** an LLM stating something plausible-sounding that isn't in the sources (or isn't true) — the key risk when answers report financial figures.
- **LangGraph:** a library for building LLM applications as explicit state machines — nodes (steps) connected by edges (transitions) operating on shared state.
- **ETL (Extract, Transform, Load):** the classic data-pipeline pattern — read raw data, reshape it, store it. Here: read PDFs, chunk + embed, load into Qdrant.
- **OCR (Optical Character Recognition):** turning images of text (scanned pages) into machine-readable text. Unstructured's `hi_res` path can invoke it; this project sets no explicit OCR options.
- **Recall@K:** of all the truly relevant chunks for a query, the fraction that appear in the top K results.
- **MRR (Mean Reciprocal Rank):** across queries, the average of 1 ÷ (rank of the first correct result). 1.0 means the right chunk is always first.
- **nDCG (normalized Discounted Cumulative Gain):** a ranking-quality score from 0 to 1 that rewards putting the most relevant results nearest the top, with graded (not just yes/no) relevance.

---

## Audit Evidence Index

Every file inspected for this audit, with the major conclusions drawn from each:

| File | Major conclusions |
|---|---|
| `README.md` | Project framing, milestone log, and the 0.816 table (single-query score; OpenAI lane 0.0000 "Dummy State"). Stale: roadmap lists the already-shipped OpenAI-embedding fix; ingestion path says `FY2025`; reasoning layer listed as future though it now exists. |
| `main.py` | Entry point; wires settings → Qdrant → Ollama → tools → LangGraph → interactive loop. `--model` CLI override. Hardcodes `top_k=5` (ignores `RETRIEVAL_TOP_K`). Uses `settings.QDRANT_COLLECTION` (→ `financial_reports`). |
| `requirements.txt` | Missing `unstructured`, `fastembed`, `openai` (all imported by code); declares `langchain-community` and `pydantic` which are never directly imported; broad version ranges; no lock file. |
| `docker-compose.yml` | Two services: Qdrant (used) and Phoenix (unused by code); deprecated `version:` key; `:latest` tags; no app container. |
| `.env.example` | Missing required `OPENAI_API_KEY`; `QDRANT_COLLECTION=aequitas_documents` contradicts code default and would break retrieval; documents three settings the code never uses. |
| `.env` (key names only — values not inspected) | Contains `TAVILY_API_KEY`, `QDRANT_HOST`, `QDRANT_PORT`, `OPENAI_API_KEY`; no `QDRANT_COLLECTION`, so the code default `financial_reports` applies. Not tracked by git (verified via `git ls-files`). |
| `.gitignore` | Stray literal `EOF` line 13; ignores `data/raw/`, `data/gold/`, `qdrant_storage/`, `.env`, `venv/`, `mlx_models/`; does **not** ignore `data/silver/`, `data/raw_backup/` (hence `?? data/` in git status). |
| `.docker/Dockerfile.agent` | Tracked in git but completely empty; referenced by nothing. |
| `config/settings.py` | `Settings` class; all env reads with defaults; `QDRANT_COLLECTION` default `financial_reports`; temperature default 0.7; `QDRANT_PATH`/`RETRIEVAL_TOP_K`/`VECTOR_SIZE` are dead settings; `validate()` only warns about Tavily. |
| `config/prompts.py` | Four prompt templates including citation and refinement prompts — **never imported anywhere** (grep-verified); dead code. |
| `src/core/database.py` | `QdrantDatabase` wrapper; `search()` supports named-vector tuples via `query_points(..., using=...)`; `create_collection`/`upsert_vectors` methods bypassed by ingestion (which uses the raw client); all errors swallowed into logs + empty returns. |
| `src/core/embeddings.py` | `FastEmbedWrapper` — BGE-small query embedding returning `("local_bge", vector)`; the agent's only query-embedding path. |
| `src/core/models.py` | `LLMModels` ChatOllama factory; only `get_default_model()` is used; `get_llama3`/`get_mistral`/`get_llm`/`EmbeddingModel` are dead code; docstring falsely says "placeholder implementations". |
| `src/ingestion/ingest_docs.py` | The whole ingestion pipeline. Confirms: `partition_pdf` + `strategy="hi_res"` + `yolox` + `chunking_strategy="by_title"` + `max_characters=2000`; no `infer_table_structure`, no OCR options, no overlap. Destructive `recreate_collection` on `financial_reports`; dual named-vector points; folder-derived company/year metadata; real OpenAI embedding calls; **confirmed element/embedding zip misalignment bug**; hardcoded host/port/path/collection. |
| `src/reasoning/state.py` | `AgentState` TypedDict + `create_initial_state()`; iteration counters for loop control. |
| `src/reasoning/nodes.py` | Keyword-heuristic router (not LLM); retrieval/search nodes; `generate_answer_node` builds context (2,000-char truncation) and uses an **inline** prompt — not `config/prompts.py`; iteration cap forces generation. |
| `src/reasoning/graph.py` | `ReasoningGraph` builds/compiles the LangGraph; conditional edges with fallback-to-generate for missing tools; `generate → END`; deterministic workflow, not an autonomous agent. |
| `src/reasoning/tools.py` | `TavilySearchTool` (optional) and `LocalRetrievalTool` (returns only `payload["text"]`, discarding source/page/score — why citations are impossible today); unused `create_tools` factory with a fourth collection-name default (`"documents"`). |
| `src/evaluation/metrics.py`, `src/evaluation/run_evals.py` | Both **0 bytes**. The evaluation package is an empty placeholder. |
| `tests/test_retrieval.py` | Manual dual-model comparison script: hardcoded query, prints scores/excerpts, **no assertions**, function name not pytest-collectable, requires live Qdrant + OpenAI key, and queries the stale `cibc_reports` collection. |
| All `__init__.py` files (`src`, `src/*`, `config`) | All empty (0 bytes) — plain package markers. |
| `data/` (structure only) | `raw/cibc/FY2024/` holds 5 PDFs (live corpus); `raw_backup/cibc/FY2025/` holds 4 PDFs (previous corpus); `silver/` and `gold/` exist and are empty; layout matches the `{company}/FY{year}` convention. |
| `qdrant_storage/collections/` (config.json + directory listing only) | Two collections exist: `cibc_reports` (23 MB) and `financial_reports` (21 MB), both with identical named-vector schemas (`local_bge` 384 + `openai` 1536, cosine) — physical proof of both the mismatch and that ingestion ran. |
| `git ls-files`, `git log`, `git branch` | `.env` and `data/` not tracked; history shows the progression (scaffold `f5f8ecf` → core logic `a78244d` → hi_res + named vectors `4744f7a` → real OpenAI embeddings `cb4300b` → RAG wired end-to-end `355ebf7` → `--model` flag `65093e5`); current branch `rebuild/architecture-audit`. |
| `venv/bin/pip list`, `pytest --collect-only` (executed, not just read) | Empirical confirmation: pytest collects **zero** tests; `unstructured 0.18.31`, `fastembed 0.7.4`, `openai 2.16.0` are installed locally despite being absent from `requirements.txt`; resolved versions of the broad ranges (langchain 0.3.27, langgraph 0.6.11, qdrant-client 1.16.2). |
| Project-wide greps | No `infer_table_structure`, no `CompositeElement` in code, no Phoenix/Arize/OpenTelemetry references, no `Filter` usage, no imports of `config.prompts`; complete inventory of collection-name references and TODO/placeholder markers. |
