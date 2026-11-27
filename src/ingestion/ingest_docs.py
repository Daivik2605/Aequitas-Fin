"""
Aequitas-Fin — document ingestion.

Turns financial PDFs into searchable vectors, twice over, in a way that is
safe to re-run.

Pipeline, per document:

    PDF
      -> partition_pdf(hi_res + YOLOX)    detect page layout, recognise tables
      -> chunking (by_title, 2000 chars)  group elements into semantic chunks
      -> _chunk_text()                    tables -> Markdown, prose -> as-is
      -> BGE + OpenAI embeddings          two vectors for the SAME text
      -> _point_id()                      content-addressed identity
      -> upsert                           write what should exist
      -> delete(filter)                   remove what should not

WHY TWO EMBEDDING MODELS AT ONCE
Both vectors attach to the same point ID and the same payload, so a local
model and a cloud model can be compared over provably identical chunks. If
they lived in two collections, a measured difference could come from the
embedding model OR from the two corpora being chunked differently — the
comparison would not be interpretable. One collection with named vectors
holds everything constant except the variable under test.

LIBRARIES AND WHAT EACH ONE IS FOR
    unstructured    PDF layout analysis + table-structure recognition.
                    The only reason this project can read financial tables.
    fastembed       Local BGE embeddings via ONNX. Runs on CPU / Apple
                    Silicon, no API call, no data leaving the machine.
    openai          Cloud embeddings (text-embedding-3-small). The paid
                    baseline the local model is measured against.
    qdrant-client   Vector database. Provides named vectors, payload
                    filtering, and HNSW approximate nearest-neighbour search.
    python-dotenv   Loads OPENAI_API_KEY and connection settings from .env
                    so secrets never live in source.
"""

import os
import uuid
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import models

from src.core.database import QdrantDatabase
from src.core.tokens import (
    BGE_MAX_TOKENS,
    count_tokens,
    split_long_text,
    summarise_tokens,
)
from config.settings import settings

# Reads .env into os.environ. Must run before anything reads OPENAI_API_KEY.
load_dotenv()


# ---------------------------------------------------------------------------
# Point identity
# ---------------------------------------------------------------------------

def _point_id(source: str, page_number, text: str) -> str:
    """
    Content-addressed point ID.

    Identity derives from what a chunk *is* — its source document, the page
    it came from, and its text — rather than its index in a list. A chunker
    change therefore produces new IDs for changed content and identical IDs
    for unchanged content, instead of silently rewriting whatever happened
    to occupy position N.

    Position is deliberately excluded so unchanged text keeps a stable ID
    across pipeline changes. The trade-off: byte-identical text on the same
    page collapses to a single point. Callers count collapses rather than
    letting them pass unnoticed.

    uuid5 (not uuid4) because it is DETERMINISTIC: the same input always
    produces the same UUID, which is what makes re-ingestion a no-op for
    unchanged content. Qdrant only accepts integers or UUIDs as point IDs,
    which is why this returns a UUID rather than a raw hash.
    """
    canonical = f"{source}|{page_number}|{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical))


# ---------------------------------------------------------------------------
# Embedding lanes — the same text, two vector spaces
# ---------------------------------------------------------------------------

def get_openai_embeddings(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Cloud lane: 1536-dimension vectors from text-embedding-3-small.

    Batched because one HTTP request per chunk would be thousands of
    round-trips dominated by network latency, and would hit rate limits
    immediately. 100 is conservative — the binding constraint is the
    per-request token limit, not the item count.

    KNOWN GAP: no retry, no backoff, no handling of 429. An error here
    loses the whole document, including batches that already succeeded.
    """
    oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = oa_client.embeddings.create(input=batch, model="text-embedding-3-small")
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


@lru_cache(maxsize=1)
def _bge_model() -> TextEmbedding:
    """Load the local embedding model once per process."""
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def get_bge_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Local lane: 384-dimension vectors from BAAI/bge-small-en-v1.5.

    Free, private, and fast enough on Apple Silicon. Its limit is a 512-token
    maximum sequence length — longer chunks are TRUNCATED SILENTLY, with no
    error. OpenAI's model accepts 8191 tokens, so on long chunks the two
    lanes are not seeing the same input. That asymmetry is a confound in any
    local-vs-cloud comparison and has to be measured and disclosed.

    The model is cached module-wide: loading it is expensive, and the old
    code constructed it inside this function, reloading it once per PDF.
    """
    return list(_bge_model().embed(texts))


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def process_cibc_reports(data_dir: str):
    # -- 1. Connect ---------------------------------------------------------
    # Host and port come from settings (env-driven) rather than literals, so
    # the same code runs against a local container or a remote Qdrant.
    db = QdrantDatabase(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    client = db.get_client()

    # -- 2. Collection setup ------------------------------------------------
    # Created once, on first run only. Two NAMED VECTOR spaces live on every
    # point: retrieval picks one at query time with `using=`.
    #   local_bge  384 dims  — must match the BGE model's output width
    #   openai    1536 dims  — must match text-embedding-3-small
    # COSINE distance because text embeddings encode meaning in direction,
    # not magnitude: a long passage and a short one on the same topic should
    # score as similar.
    if not db.collection_exists(settings.QDRANT_COLLECTION):
        print("Creating Qdrant collection for Named Vectors...")
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                "local_bge": models.VectorParams(size=384, distance=models.Distance.COSINE),
                "openai": models.VectorParams(size=1536, distance=models.Distance.COSINE),
            },
        )
        # Payload indexes. Without them, filtering on a payload field scans
        # and parses EVERY point in the collection. Needed by the prune step
        # below (filters on source once per document) and by query-time
        # filtering by company / fiscal year.
        # KEYWORD schema = exact-match categorical strings, which is what a
        # filename, a company name and a year are.
        # Built here because adding an index to a populated collection is a
        # migration; doing it at creation means there is never one to run.
        for field in ("metadata.source", "metadata.company", "metadata.fiscal_year"):
            client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        print("Created payload indexes for source / company / fiscal_year")
    else:
        print(f"Collection '{settings.QDRANT_COLLECTION}' already exists. skip creation, will upsert into it")

    # -- 3. Discover documents ----------------------------------------------
    # glob("**/*.pdf") walks every subdirectory and matches any .pdf.
    # sorted() because glob returns filesystem order, which varies by machine
    # and changes as files are added — sorting makes two runs' logs diffable.
    raw_path = Path(data_dir)
    pdf_files = sorted(raw_path.glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {data_dir}. Check your folders!")
        return

    for pdf_file in pdf_files:
        print(f"\nPartitioning: {pdf_file.name} using hi_res...")

        # -- 4. Parse -------------------------------------------------------
        # strategy="hi_res" renders each page to an image and runs a vision
        # model over it, instead of pulling the PDF's raw text layer. Roughly
        # an order of magnitude slower, but the text layer reads a financial
        # table in glyph-storage order and destroys row/column alignment.
        #
        # hi_res_model_name="yolox" is the layout DETECTION model: it finds
        # where titles, paragraphs and tables are on the page.
        #
        # infer_table_structure=True is a SECOND model pass that recovers the
        # cells inside each detected table and emits HTML. Without it you know
        # a table exists but still get flattened text — paying hi_res prices
        # for none of the table benefit.
        #
        # chunking_strategy="by_title" starts a new chunk at each detected
        # heading, so boundaries follow the document's own structure instead
        # of falling mid-sentence at an arbitrary character count.
        # max_characters caps chunk size so embeddings stay within model
        # input limits and each chunk is about one thing.
        elements = partition_pdf(
            filename=str(pdf_file),
            strategy="hi_res",
            hi_res_model_name="yolox",
            chunking_strategy="by_title",
            max_characters=2000,
            # infer_table_structure is OFF by measurement, not by default.
            # scripts/parser_probe.py scored four configurations on whether a
            # row label survives next to its value: hi_res alone scored 7/8,
            # hi_res + table structure also 7/8 — for 58% more parse time. The
            # table model's HTML collapses whole columns into single cells, so
            # rendering it as Markdown scored WORSE (6/8) and lost one value
            # outright. See docs/DECISIONS.md.
        )

        # -- 5. Choose the text for each chunk ------------------------------
        # Filter BEFORE zipping with embeddings. Filtering afterwards is the
        # off-by-one that silently attached vectors to the wrong chunks.
        valid_elements = [el for el in elements if el.text and el.text.strip()]

        def _chunk_texts(el) -> list[str]:
            # Returns a LIST because a chunk over the local model's 512-token
            # limit is split rather than silently truncated.
            #
            # Tables use unstructured's linearised .text like everything else.
            # The structure-aware HTML path was tried and measured against it
            # (scripts/parser_probe.py) and scored worse: the table model
            # collapses entire columns into single cells, so the rendered
            # Markdown looked structured while being wrong. .text keeps each
            # row label beside its own values, which is what actually matters.
            return split_long_text(el.text)

        # One element can now yield several chunks, so carry the pairing
        # explicitly instead of relying on two lists staying the same length.
        # This is the same invariant the old zip bug violated, made structural
        # rather than assumed.
        pairs = [(el, text) for el in valid_elements for text in _chunk_texts(el)]
        texts = [text for _, text in pairs]

        # -- 6. Embed the same text twice -----------------------------------
        print(f"Generating BGE embeddings for {len(texts)} chunks...")
        bge_embeddings = get_bge_embeddings(texts)
        print(f" Generating OpenAI embeddings for {len(texts)} chunks (batches of 100)...")
        openai_embeddings = get_openai_embeddings(texts)

        # Makes the alignment invariant LOUD. The original bug here failed
        # silently; an assertion turns that class of defect into a crash.
        assert len(pairs) == len(bge_embeddings) == len(openai_embeddings)

        # Truncation report. Printed per document so it is visible while it
        # happens, not discovered later as unexplained holes in retrieval.
        token_counts = [count_tokens(text) for text in texts]
        stats = summarise_tokens(token_counts)
        print(
            f"  tokens: median={stats['median']} p95={stats['p95']} "
            f"max={stats['max']} | over {BGE_MAX_TOKENS}: {stats['over_limit']}"
            f"/{stats['n']} chunks"
        )

        # -- 7. Derive metadata from the path -------------------------------
        # Layout is data/raw/{company}/FY{year}/file.pdf — deriving company
        # and year from the directory structure avoids parsing them out of
        # document text, which is far less reliable.
        fiscal_year_folder = pdf_file.parent.name          # e.g. "FY2025"
        year = fiscal_year_folder.replace("FY", "")        # e.g. "2025"
        company = pdf_file.parent.parent.name.lower()      # e.g. "cibc"

        # -- 8. Build points ------------------------------------------------
        # One point = one chunk = both vectors + the text + its provenance.
        # page_number and source are what let a generated answer cite where
        # the figure came from.
        points, seen, collapsed = [], set(), 0
        for (element, chunk_text), bge_vec, oai_vec, n_tokens in zip(
            pairs, bge_embeddings, openai_embeddings, token_counts
        ):
            el_dict = element.to_dict()
            metadata = el_dict.get("metadata", {})
            page_number = metadata.get("page_number")

            point_id = _point_id(pdf_file.name, page_number, chunk_text)
            # Identical text on the same page maps to the same ID. Count it
            # rather than letting the collapse pass unnoticed.
            if point_id in seen:
                collapsed += 1
                continue
            seen.add(point_id)

            points.append(models.PointStruct(
                id=point_id,
                vector={"local_bge": bge_vec.tolist(), "openai": oai_vec},
                payload={
                    "text": chunk_text,
                    "metadata": {
                        "source": pdf_file.name,
                        "company": company,
                        "fiscal_year": year,
                        "element_type": el_dict.get("type"),
                        "page_number": page_number,
                        # Stored so retrieval quality can later be analysed by
                        # chunk size — e.g. whether short heading-like chunks
                        # crowd out substantive ones in top-k.
                        "token_count": n_tokens,
                        "char_count": len(chunk_text),
                    },
                },
            ))

        if collapsed:
            print(f"  collapsed {collapsed} duplicate chunk(s) with identical text on the same page")

        if not points:
            print(f"  no usable chunks from {pdf_file.name}; skipping")
            continue

        # -- 9. Write: upsert, then prune -----------------------------------
        # upsert = update-or-insert, keyed on point ID. Idempotent: running it
        # five times leaves the same state as running it once.
        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)

        # Upsert can only add or overwrite; it has no opinion about points
        # that exist but were not in this batch. This delete removes them:
        #   must      -> scope to this document only
        #   must_not  -> spare the IDs this run just wrote
        # i.e. "everything from this file that I did not just produce".
        # Together, upsert + prune make re-ingesting a document a REPLACE.
        #
        # ORDER MATTERS. Delete-then-insert leaves a window where the document
        # is absent; a crash there destroys data with nothing to replace it,
        # and retrieval silently stops finding that source. Upsert-then-prune
        # fails toward EXTRA stale points, which a re-run cleans up. When an
        # operation cannot be atomic, prefer leaving too much over too little.
        pruned = client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=pdf_file.name),
                    )],
                    must_not=[models.HasIdCondition(has_id=[p.id for p in points])],
                )
            ),
        )
        print(f"Indexed {len(points)} points from {pdf_file.name} (stale points pruned: {pruned.status})")


if __name__ == "__main__":
    process_cibc_reports("data/raw/cibc")
