"""Token-aware recursive text chunker with configurable overlap.

Strategy
--------
1. Split on double-newlines (paragraphs), then single newlines, then spaces —
   recursively until every piece is ≤ ``chunk_size`` tokens.
2. Slide a window of ``overlap`` tokens between consecutive chunks so that
   retrieval can recover context that spans a chunk boundary.

Token counting uses a simple whitespace approximation (≈ character / 4)
so the chunker has *zero* external dependencies — no tiktoken, no model call.
This is intentionally conservative: real token counts will be slightly higher,
keeping chunks safely under the Voyage 3 input limit (32 K tokens).
"""

from __future__ import annotations

from dataclasses import dataclass

_SEPARATORS = ["\n\n", "\n", " "]


@dataclass(frozen=True)
class Chunk:
    """A single text chunk ready for embedding."""

    text: str
    index: int


def _approx_tokens(text: str) -> int:
    """Approximate token count: characters / 4."""
    return max(1, len(text) // 4)


def _split_text(text: str, separators: list[str]) -> list[str]:
    """Recursively split *text* using the first matching separator."""
    if not separators:
        # No more separators — return the text as-is (atomic piece).
        return [text] if text.strip() else []
    sep, *rest = separators
    parts = text.split(sep)
    if len(parts) == 1:
        # Separator not found — try the next one.
        return _split_text(text, rest)
    result = []
    for p in parts:
        if p.strip():
            result.extend(_split_text(p, rest))
    return result


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Split *text* into overlapping token-bounded :class:`Chunk` objects.

    Args:
        text: Raw document text.
        chunk_size: Target max tokens per chunk.
        overlap: Number of tokens to repeat at the start of the next chunk.

    Returns:
        Ordered list of :class:`Chunk` instances.
    """
    if not text.strip():
        return []

    pieces = _split_text(text.strip(), _SEPARATORS)
    if not pieces:
        return []

    # Merge small pieces into chunks up to chunk_size tokens.
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _approx_tokens(piece)
        if current_tokens + piece_tokens > chunk_size and current_parts:
            chunks.append(" ".join(current_parts))
            # Carry-over overlap: drop leading parts until within budget.
            while current_parts and current_tokens > overlap:
                removed = current_parts.pop(0)
                current_tokens -= _approx_tokens(removed)
        current_parts.append(piece)
        current_tokens += piece_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [Chunk(text=c.strip(), index=i) for i, c in enumerate(chunks) if c.strip()]
