# Decisions

Each entry records a question, the measurement that answered it, and the
resulting choice. A decision without a number behind it is an opinion.

---

## D1 — PDF parsing strategy

**Date:** 2026-09-15
**Question:** Which parsing configuration best preserves the association
between a row label and its value? In a financial document, `4,946` is not an
answer; `Balance at end of period: 4,946` is.

**Measurement:** `scripts/parser_probe.py` — 8 known (label, value) pairs read
off pages 12 and 14 of `news_release.pdf`. A needle counts as *associated*
when the value appears in the extracted text with its label within 300
characters.

| config | parse time | present | associated |
| --- | --- | --- | --- |
| `fast` | 2s | 8/8 | 4/8 |
| `hi_res` | 19s | 8/8 | **7/8** |
| `hi_res` + `infer_table_structure` | 30s | 8/8 | 7/8 |
| `hi_res` + table structure, rendered as Markdown | – | 7/8 | 6/8 |

**Decisions:**

1. **Use `hi_res`.** It nearly doubles label-value association over `fast`
   (7/8 vs 4/8) and is worth roughly 10× the parse time. `fast` reads the
   PDF's text layer in storage order, which scatters table values away from
   their row labels.
2. **Leave `infer_table_structure` off.** It scores identically to `hi_res`
   alone on this metric while adding 58% to parse time. The benefit it is
   supposed to provide does not materialise on these documents.
3. **Do not render table HTML as Markdown.** It scored *worse* (6/8) and lost
   one value entirely. The table-structure model collapses whole columns into
   single cells carrying `rowspan="4"` — for example, every row label in one
   cell and every October-2024 value in another. Rendering that faithfully
   produces output that looks structured and is wrong, which is more dangerous
   than output that looks unstructured. unstructured's linearised `.text`
   keeps each row label beside its own values.

**Known limitation:** all `hi_res` configurations miss
`Corporate and Other -> 285` on the segmented-results table, whose multi-level
header the layout model reads column-wise. This is a document-layout limit,
not a configuration one — it is identical across configs.

**Reversal:** this replaced an earlier, unmeasured decision to enable
`infer_table_structure` and render tables as Markdown. The hypothesis was that
structure-aware HTML would preserve column meaning. The probe disproved it.
The code was written before the input was inspected — the correct order is to
look at what a library actually emits on your real documents first.
