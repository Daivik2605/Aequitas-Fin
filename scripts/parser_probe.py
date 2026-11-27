"""
Parser comparison probe.

THE QUESTION: which PDF parsing configuration best preserves the association
between a row label and its number? That association IS the answer in a
financial document — "4,946" alone is not an answer, "Balance at end of
period: 4,946" is.

THE METHOD: a needle test. We know specific (label, value) pairs that exist
in these pages because we read them off the source. For each parser config we
check two things:
  presence    - does the value appear in the extracted text at all?
  association - does its label appear within PROXIMITY characters of it?

Association is the metric that matters. A parser can emit every number in the
document and still be useless if the labels are somewhere else entirely.

No ground truth authoring required, no embeddings, no database. Runs in a
couple of minutes and gives an objective basis for a decision that would
otherwise be made on vibes.

Re-run this after any parsing change to see whether it helped or hurt.
"""

import argparse
import time
from datetime import date
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from unstructured.partition.pdf import partition_pdf

from src.core.tables import html_table_to_markdown

PROXIMITY = 300  # chars between label and value to count as "associated"

# Verified by reading the source pages. Page numbers are 1-indexed.
NEEDLES = [
    # page 12 - preferred shares continuity table
    (12, "Balance at beginning of period", "4,949"),
    (12, "Issue of preferred shares", "500"),
    (12, "Redemption of preferred shares", "(650)"),
    (12, "Treasury shares", "(3)"),
    (12, "Balance at end of period", "4,946"),
    # page 14 - Q4 2024 segmented results
    (14, "CIBC", "6,617"),
    (14, "CIBC", "1,882"),
    (14, "Corporate and Other", "285"),
]

CONFIGS = {
    "fast": dict(strategy="fast"),
    "hi_res": dict(strategy="hi_res", hi_res_model_name="yolox"),
    "hi_res+tables": dict(
        strategy="hi_res", hi_res_model_name="yolox", infer_table_structure=True
    ),
}


def extract_pages(src: Path, pages_1indexed, dest: Path) -> None:
    reader = PdfReader(str(src))
    writer = PdfWriter()
    for p in pages_1indexed:
        writer.add_page(reader.pages[p - 1])
    with open(dest, "wb") as fh:
        writer.write(fh)


def score(text: str, needles) -> dict:
    """Presence and label-value association for one extracted text blob."""
    present, associated, misses = 0, 0, []
    for _page, label, value in needles:
        if value not in text:
            misses.append(f"{label} -> {value} (value absent)")
            continue
        present += 1
        ok = False
        start = 0
        while (idx := text.find(value, start)) != -1:
            window = text[max(0, idx - PROXIMITY): idx + PROXIMITY]
            if label.lower() in window.lower():
                ok = True
                break
            start = idx + 1
        if ok:
            associated += 1
        else:
            misses.append(f"{label} -> {value} (value found, label far away)")
    return {
        "present": present,
        "associated": associated,
        "total": len(needles),
        "misses": misses,
    }


def run(pdf: Path, pages, out: Path) -> None:
    probe = Path("/tmp/parser_probe.pdf")
    extract_pages(pdf, pages, probe)
    needles = [n for n in NEEDLES if n[0] in pages]

    lines = [
        f"# Parser probe - {date.today()}",
        "",
        f"Source: `{pdf.name}` pages {pages}  |  needles: {len(needles)}  "
        f"|  proximity window: {PROXIMITY} chars",
        "",
        "**Association** is the metric that matters: the value found *and* its row",
        "label within the window. Presence alone means the number survived but may",
        "have lost what it refers to.",
        "",
        "| config | time | elements | tables | present | associated |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    details = []

    for name, kwargs in CONFIGS.items():
        t0 = time.perf_counter()
        els = partition_pdf(filename=str(probe), **kwargs)
        elapsed = time.perf_counter() - t0

        tables = [e for e in els if type(e).__name__ == "Table"]
        text = "\n".join(e.text or "" for e in els)
        s = score(text, needles)

        lines.append(
            f"| {name} | {elapsed:.0f}s | {len(els)} | {len(tables)} | "
            f"{s['present']}/{s['total']} | **{s['associated']}/{s['total']}** |"
        )
        details.append((name, s, tables))

        # The Markdown path gets scored as its own variant, so the decision to
        # keep or drop it rests on the same measurement as everything else.
        if name == "hi_res+tables" and tables:
            md_text = "\n".join(
                html_table_to_markdown(getattr(e.metadata, "text_as_html", "") or "")
                or (e.text or "")
                for e in els
            )
            sm = score(md_text, needles)
            lines.append(
                f"| hi_res+tables (markdown) | - | {len(els)} | {len(tables)} | "
                f"{sm['present']}/{sm['total']} | **{sm['associated']}/{sm['total']}** |"
            )
            details.append(("hi_res+tables (markdown)", sm, []))

    lines.append("")
    for name, s, _ in details:
        lines.append(f"### {name}")
        if s["misses"]:
            lines.extend(f"- MISS: {m}" for m in s["misses"])
        else:
            lines.append("- all needles associated")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    out.write_text(report)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/raw/cibc/FY2024/news_release.pdf")
    ap.add_argument("--pages", default="12,14")
    ap.add_argument("--out", default="docs/PARSER_PROBE.md")
    a = ap.parse_args()
    run(Path(a.pdf), [int(p) for p in a.pages.split(",")], Path(a.out))
