"""
Single-page inspector.

Answers "what did the parser actually see on this page?" — element by element,
with type, page, token count, and text. This is the microscope: when retrieval
misses something, look at the page it should have come from and find out
whether the text was ever extracted in the first place.

Usage:
    python3 scripts/page_inspect.py --page 14
    python3 scripts/page_inspect.py --page 14 --strategy fast
    python3 scripts/page_inspect.py --page 14 --tables          # table HTML too
    python3 scripts/page_inspect.py --page 14 --find 1,882      # locate a value
    python3 scripts/page_inspect.py --pdf data/raw/cibc/FY2024/mda.pdf --page 7
"""

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from unstructured.partition.pdf import partition_pdf

from src.core.tokens import count_tokens


def one_page(src: Path, page: int, dest: Path) -> None:
    reader = PdfReader(str(src))
    writer = PdfWriter()
    writer.add_page(reader.pages[page - 1])
    with open(dest, "wb") as fh:
        writer.write(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/raw/cibc/FY2024/news_release.pdf")
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--strategy", default="hi_res", choices=["fast", "hi_res", "ocr_only", "auto"])
    ap.add_argument("--model", default="yolox", help="layout model for hi_res")
    ap.add_argument("--tables", action="store_true", help="also dump table HTML")
    ap.add_argument("--chunk", action="store_true", help="apply by_title chunking")
    ap.add_argument("--find", default=None, help="highlight elements containing this string")
    ap.add_argument("--full", action="store_true", help="print whole text, not first 400 chars")
    a = ap.parse_args()

    probe = Path("/tmp/page_inspect.pdf")
    one_page(Path(a.pdf), a.page, probe)

    kwargs = {"filename": str(probe), "strategy": a.strategy}
    if a.strategy == "hi_res":
        kwargs["hi_res_model_name"] = a.model
        if a.tables:
            kwargs["infer_table_structure"] = True
    if a.chunk:
        kwargs["chunking_strategy"] = "by_title"
        kwargs["max_characters"] = 2000

    els = partition_pdf(**kwargs)

    print(f"{Path(a.pdf).name} page {a.page} | strategy={a.strategy}"
          f"{' +tables' if a.tables else ''}{' +chunking' if a.chunk else ''}")
    print(f"{len(els)} elements\n")

    for i, el in enumerate(els):
        kind = type(el).__name__
        text = el.text or ""
        hit = a.find and a.find in text
        marker = "  <<< MATCH" if hit else ""
        print(f"[{i}] {kind} | {count_tokens(text)} tok | {len(text)} chars{marker}")
        print(text if a.full or hit else text[:400])
        if a.tables and (html := getattr(el.metadata, "text_as_html", None)):
            print(f"    -- text_as_html: {html.count('<tr')} rows, {len(html)} chars --")
            print("   ", html[:400])
        print()

    if a.find:
        n = sum(1 for el in els if a.find in (el.text or ""))
        print(f"'{a.find}' found in {n} of {len(els)} elements")


if __name__ == "__main__":
    main()
