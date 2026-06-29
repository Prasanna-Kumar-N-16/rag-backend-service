"""Cohere rerank-3 service layer.

Takes the fused candidates from :mod:`app.retrieval.hybrid` and uses Cohere's
dedicated reranking model to re-score them with cross-encoder quality,
returning only the top-k most relevant chunks to the generation step.

This is Stage 2 of the multi-stage retrieval pipeline:
  hybrid (vector + lexical + RRF) → **rerank** → top-k → generate
"""

from __future__ import annotations

from dataclasses import dataclass

import cohere

from app.config import Settings
from app.logging import get_logger
from app.retrieval.hybrid import FusedResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class RankedResult:
    """A reranked chunk with its Cohere relevance score."""

    id: str
    source_key: str
    chunk_index: int
    content: str
    relevance_score: float


class Reranker:
    """Wraps the Cohere rerank API for synchronous-in-async use."""

    def __init__(self, settings: Settings) -> None:
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required for reranking.")
        self._model = settings.rerank_model
        self._top_k = settings.rerank_top_k
        # cohere.AsyncClientV2 is the current async client (cohere ≥ 5.x).
        self._client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)

    async def rerank(
        self,
        query: str,
        candidates: list[FusedResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """Rerank *candidates* for *query* and return the top-k results.

        Args:
            query: The original user query.
            candidates: Fused candidates from the hybrid retrieval stage.
            top_k: Override the configured top-k (uses ``settings.rerank_top_k``
                if ``None``).

        Returns:
            List of :class:`RankedResult` ordered by descending relevance score,
            truncated to ``top_k`` items.
        """
        if not candidates:
            return []

        k = top_k if top_k is not None else self._top_k
        documents = [c.content for c in candidates]

        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(k, len(candidates)),
        )

        results: list[RankedResult] = []
        for hit in response.results:
            candidate = candidates[hit.index]
            results.append(
                RankedResult(
                    id=candidate.id,
                    source_key=candidate.source_key,
                    chunk_index=candidate.chunk_index,
                    content=candidate.content,
                    relevance_score=float(hit.relevance_score),
                )
            )

        logger.info(
            "rerank_done",
            model=self._model,
            candidates_in=len(candidates),
            top_k=k,
            returned=len(results),
        )
        return results
