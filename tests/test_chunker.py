"""Unit tests for the text chunker — no external dependencies."""

from __future__ import annotations

from app.indexing.chunker import Chunk, chunk_text


def test_empty_string_returns_no_chunks() -> None:
    assert chunk_text("") == []


def test_single_short_text_is_one_chunk() -> None:
    result = chunk_text("Hello world.", chunk_size=400, overlap=80)
    assert len(result) == 1
    assert result[0].text == "Hello world."
    assert result[0].index == 0


def test_chunks_are_indexed_sequentially() -> None:
    # Build a text long enough to force multiple chunks.
    long_text = " ".join(["word"] * 800)
    chunks = chunk_text(long_text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_no_empty_chunks_produced() -> None:
    text = "\n\n".join(["short paragraph"] * 20)
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_overlap_produces_multiple_chunks_from_long_text() -> None:
    """A long text with overlap produces more chunks than without overlap.

    The chunker carry-over means the combined token count across all chunk
    boundaries is higher than the raw text token count — each chunk boundary
    re-emits ``overlap`` tokens.  We verify structural soundness: every word
    in every chunk came from the original text (no hallucination) and consecutive
    chunks are not identical.
    """
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    word_set = set(words)

    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2, "Expected at least 2 chunks for a long text"

    for chunk in chunks:
        for w in chunk.text.split():
            assert w in word_set, f"Chunk contains word not in original text: {w}"

    # Consecutive chunks must not be identical
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.text != b.text


def test_returns_chunk_dataclass_instances() -> None:
    chunks = chunk_text("Some text here.", chunk_size=400, overlap=80)
    for chunk in chunks:
        assert isinstance(chunk, Chunk)


def test_whitespace_only_text_returns_no_chunks() -> None:
    assert chunk_text("   \n\n   \t  ") == []
