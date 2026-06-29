"""Hybrid retrieval: vector + lexical search fused via Reciprocal Rank Fusion.

Pipeline
--------
1. Run pgvector cosine search (top-N candidates) in parallel with Postgres
   full-text search (top-N candidates).
2. Merge both ranked lists with **Reciprocal Rank Fusion** (RRF).
   RRF score = Σ 1 / (k + rank_i) for each list that contains the document.
   ``k=60`` is the standard value from the original Cormack et al. paper —
   it dampens the influence of very highly-ranked documents so that consensus
   across lists matters more than dominance in one.
3. Return the top-N merged candidates (before reranking).

References
----------
Cormack, Clarke, Buettcher (2009) "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.indexing.embedder import Embedder
from app.logging import get_logger
from app.retrieval.vector_store import SearchResult, lexical_search, vector_search

logger = get_logger(__name__)

_RRF_K = 60  # standard dampening constant


@dataclass(frozen=True)
class FusedResult:
    """A candidate after RRF fusion, carrying the original chunk data."""

    id: str
    source_key: str
    chunk_index: int
    content: str
    rrf_score: float


def reciprocal_rank_fusion(
    *ranked_lists: list[SearchResult],
    k: int = _RRF_K,
) -> list[FusedResult]:
    """Fuse an arbitrary number of ranked result lists using RRF.

    Args:
        *ranked_lists: One or more lists of :class:`SearchResult`, each already
            ordered by descending relevance (index 0 = most relevant).
        k: RRF dampening constant (default 60).

    Returns:
        List of :class:`FusedResult` sorted by descending RRF score.
        Deduplication is by ``SearchResult.id``.
    """
    scores: dict[str, float] = {}
    meta: dict[str, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
            meta.setdefault(result.id, result)

    return [
        FusedResult(
            id=doc_id,
            source_key=meta[doc_id].source_key,
            chunk_index=meta[doc_id].chunk_index,
            content=meta[doc_id].content,
            rrf_score=score,
        )
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


async def hybrid_retrieve(
    query: str,
    embedder: Embedder,
    top_n: int = 40,
) -> list[FusedResult]:
    """Run vector + lexical search in parallel and fuse with RRF.

    Args:
        query: Raw user query string.
        embedder: :class:`Embedder` instance used to embed the query.
        top_n: Candidate count for each leg and for the fused output.

    Returns:
        Up to ``top_n`` :class:`FusedResult` items ordered by RRF score.
    """
    # Embed the query and run both search legs concurrently.
    query_embeddings = await embedder.embed([query])
    query_vec = query_embeddings[0]

    vector_results, lexical_results = await asyncio.gather(
        vector_search(query_vec, top_n=top_n),
        lexical_search(query, top_n=top_n),
    )

    fused = reciprocal_rank_fusion(vector_results, lexical_results)

    logger.info(
        "hybrid_retrieve_done",
        vector_hits=len(vector_results),
        lexical_hits=len(lexical_results),
        fused=len(fused),
    )
    return fused[:top_n]
