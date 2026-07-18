"""pgvector cosine-similarity search.

Returns ranked candidates (source_key, chunk text, score) from the
``document_chunks`` table using its HNSW index.
"""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from app.db import get_pool
from app.logging import get_logger

logger = get_logger(__name__)

_VECTOR_SEARCH_SQL = """
SELECT
    id,
    source_key,
    chunk_index,
    content,
    1 - (embedding <=> $1::vector) AS score
FROM document_chunks
ORDER BY embedding <=> $1::vector
LIMIT $2
"""


@dataclass(frozen=True)
class SearchResult:
    """A single retrieved chunk with its relevance score."""

    id: str
    source_key: str
    chunk_index: int
    content: str
    score: float


async def vector_search(
    query_embedding: list[float],
    top_n: int = 40,
) -> list[SearchResult]:
    """Run a cosine-similarity nearest-neighbour search against pgvector.

    Args:
        query_embedding: Dense embedding of the user query (same dim as stored vectors).
        top_n: Maximum number of candidates to return.

    Returns:
        List of :class:`SearchResult` ordered by descending cosine similarity.
    """
    pool: asyncpg.Pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(_VECTOR_SEARCH_SQL, query_embedding, top_n)

    results = [
        SearchResult(
            id=str(row["id"]),
            source_key=str(row["source_key"]),
            chunk_index=int(row["chunk_index"]),
            content=str(row["content"]),
            score=float(row["score"]),
        )
        for row in rows
    ]
    logger.debug("vector_search_done", top_n=top_n, returned=len(results))
    return results


async def lexical_search(
    query: str,
    top_n: int = 40,
) -> list[SearchResult]:
    """Full-text search using the ``tsvector`` GIN index.

    Uses Postgres ``ts_rank_cd`` for scoring so ranking is consistent with the
    cosine path and can be fused via RRF.

    Args:
        query: Raw query text (converted to ``tsquery`` via ``plainto_tsquery``).
        top_n: Maximum number of candidates to return.

    Returns:
        List of :class:`SearchResult` ordered by descending FTS rank.
    """
    sql = """
    SELECT
        id,
        source_key,
        chunk_index,
        content,
        ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score
    FROM document_chunks
    WHERE tsv @@ plainto_tsquery('english', $1)
    ORDER BY score DESC
    LIMIT $2
    """
    pool: asyncpg.Pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, query, top_n)

    results = [
        SearchResult(
            id=str(row["id"]),
            source_key=str(row["source_key"]),
            chunk_index=int(row["chunk_index"]),
            content=str(row["content"]),
            score=float(row["score"]),
        )
        for row in rows
    ]
    logger.debug("lexical_search_done", top_n=top_n, returned=len(results))
    return results
