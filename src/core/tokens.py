"""
Token counting for the local BGE lane.

WHY THIS EXISTS
BAAI/bge-small-en-v1.5 is a BERT-family encoder with a 512-token maximum
sequence length, and fastembed truncates anything longer SILENTLY — no
error, no warning, no log line. A chunk over the limit is embedded from its
opening tokens only; everything after that never reaches the vector and is
therefore unfindable, no matter how good the query.

This matters twice over:

 1. CORRECTNESS. Markdown tables are token-dense — pipes, separators and
    comma-grouped numerals all tokenize poorly, roughly 2.8 characters per
    token against ~4.2 for prose. A 2000-character table lands near 700
    tokens, so its final rows are invisible to search. In a financial
    statement the later rows are often the ones being asked about.

 2. EXPERIMENTAL VALIDITY. text-embedding-3-small accepts 8191 tokens. On
    long chunks the two lanes are not seeing the same input — one is
    truncated, the other is not. Any local-vs-cloud comparison has to
    measure how often that happens and disclose it, or the result is
    unattributable: a cloud win could be model quality or could be the
    local model never having seen the answer.

Uses the `tokenizers` library directly (already a fastembed dependency) with
the model's own tokenizer.json, so counts match what the embedding model
actually does rather than approximating.
"""

from functools import lru_cache

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Hard ceiling: 512 positions, two of which are the [CLS] and [SEP] tokens.
BGE_MAX_TOKENS = 512

# What we actually pack to. Headroom so a split piece cannot land on the
# boundary and be truncated by one or two tokens.
BGE_SAFE_BUDGET = 480


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The `tokenizers` package is required for token counting. "
            "It ships with fastembed; try `pip install tokenizers`."
        ) from exc
    return Tokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text: str) -> int:
    """How many tokens BGE will see, special tokens included."""
    return len(_tokenizer().encode(text).ids)


def summarise_tokens(counts: list[int]) -> dict:
    """
    Distribution summary for one document's chunks.

    Reported per document during ingestion so truncation is visible while it
    is happening, rather than being discovered later as unexplained gaps in
    retrieval.
    """
    if not counts:
        return {"n": 0}
    ordered = sorted(counts)
    n = len(ordered)
    return {
        "n": n,
        "median": ordered[n // 2],
        "p95": ordered[min(n - 1, int(n * 0.95))],
        "max": ordered[-1],
        "over_limit": sum(1 for c in ordered if c > BGE_MAX_TOKENS),
    }


# Boundaries to split on, largest structural unit first. Splitting where the
# document already breaks keeps pieces readable; a mid-word cut does not.
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """
    Last-resort cut for a run of text with no usable boundary left.

    Measures the text's OWN characters-per-token ratio rather than assuming
    one. The first version of this assumed ~3 chars/token, which is right for
    prose and badly wrong for financial tables — comma-grouped numerals and
    currency symbols run closer to 2 chars/token, so the "safe" cut produced
    pieces 20% over the limit. Measuring instead of guessing is the fix, and
    the trailing loop verifies rather than trusting the estimate.
    """
    pieces: list[str] = []
    remaining = text
    while remaining:
        total = count_tokens(remaining)
        if total <= max_tokens:
            pieces.append(remaining)
            break
        ratio = len(remaining) / total          # measured, not assumed
        head = remaining[: max(1, int(max_tokens * ratio * 0.85))]
        while count_tokens(head) > max_tokens and len(head) > 1:
            head = head[: int(len(head) * 0.9)]
        pieces.append(head)
        remaining = remaining[len(head):]
    return pieces


def split_long_text(text: str, max_tokens: int = BGE_SAFE_BUDGET, _level: int = 0) -> list[str]:
    """
    Split any chunk that exceeds the local model's input limit.

    Applies to every chunk, not just tables. BGE truncates past 512 tokens
    silently, so an oversized chunk's tail is unfindable no matter how good
    the query.

    Tries progressively finer boundaries — blank lines, lines, sentences,
    words — and RECURSES into any piece that is still too large rather than
    abandoning the attempt. The earlier version checked whether every piece
    fitted and, if one did not, threw the whole split away and moved to the
    next separator. That let oversized pieces reach the crude character
    fallback, which is how four chunks landed over the limit.

    Guarantee: every returned piece is at or under max_tokens.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    if _level >= len(_SEPARATORS):
        return _hard_split(text, max_tokens)

    separator = _SEPARATORS[_level]
    parts = text.split(separator)
    if len(parts) < 2:
        return split_long_text(text, max_tokens, _level + 1)

    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(current)
            current = part
        else:
            current = candidate
    if current:
        pieces.append(current)

    # Any piece still too large goes to the next, finer boundary.
    out: list[str] = []
    for piece in pieces:
        out.extend(split_long_text(piece, max_tokens, _level + 1))
    return out
