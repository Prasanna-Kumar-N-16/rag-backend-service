"""Tests for the multi-stage retrieval pipeline.

All external I/O (pgvector pool, Cohere) is mocked — the suite runs offline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.retrieval.hybrid import FusedResult, reciprocal_rank_fusion
from app.retrieval.vector_store import SearchResult

# ── helpers ─────────────────────────────────────────────────────────────────

def _make_result(id_: str, source_key: str = "doc.txt", score: float = 1.0) -> SearchResult:
    return SearchResult(
        id=id_,
        source_key=source_key,
        chunk_index=0,
        content=f"content of {id_}",
        score=score,
    )


# ── RRF unit tests (pure, no mocks needed) ──────────────────────────────────

class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        ranked = [_make_result(f"doc{i}") for i in range(5)]
        fused = reciprocal_rank_fusion(ranked)
        assert [r.id for r in fused] == ["doc0", "doc1", "doc2", "doc3", "doc4"]

    def test_document_in_both_lists_scores_higher(self) -> None:
        """A doc in both lists outscores one that appears in only one."""
        shared = _make_result("shared")
        only_vector = _make_result("only_vector")
        only_lexical = _make_result("only_lexical")

        vector_list = [shared, only_vector]
        lexical_list = [shared, only_lexical]

        fused = reciprocal_rank_fusion(vector_list, lexical_list)
        ids = [r.id for r in fused]

        assert ids[0] == "shared", "Document in both lists should rank first"

    def test_rrf_deduplicates_by_id(self) -> None:
        doc = _make_result("dup")
        fused = reciprocal_rank_fusion([doc, doc], [doc])
        assert sum(1 for r in fused if r.id == "dup") == 1

    def test_empty_lists_return_empty(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_single_document_returns_single_result(self) -> None:
        fused = reciprocal_rank_fusion([_make_result("solo")], [])
        assert len(fused) == 1
        assert fused[0].id == "solo"

    def test_rrf_score_is_sum_of_reciprocal_ranks(self) -> None:
        """Verify the mathematical formula: score = 1/(k+1) + 1/(k+1) for rank-1 in two lists."""
        k = 60
        doc = _make_result("a")
        fused = reciprocal_rank_fusion([doc], [doc], k=k)
        expected = 2.0 / (k + 1)
        assert abs(fused[0].rrf_score - expected) < 1e-9

    def test_three_lists_accumulate_scores(self) -> None:
        doc = _make_result("triple")
        fused = reciprocal_rank_fusion([doc], [doc], [doc])
        # Score must be higher than from two lists
        fused_two = reciprocal_rank_fusion([doc], [doc])
        assert fused[0].rrf_score > fused_two[0].rrf_score

    def test_result_type_is_fused_result(self) -> None:
        fused = reciprocal_rank_fusion([_make_result("x")])
        assert all(isinstance(r, FusedResult) for r in fused)


# ── hybrid_retrieve integration test (mocked DB + Voyage) ───────────────────

class TestHybridRetrieve:
    def _make_settings(self) -> Any:
        from app.config import Settings
        return Settings(
            voyage_api_key="test",
            database_url="postgresql://x:x@localhost/test",
            embedding_dimensions=4,
        )

    @patch("app.retrieval.hybrid.lexical_search")
    @patch("app.retrieval.hybrid.vector_search")
    async def test_runs_both_legs_concurrently_and_fuses(
        self,
        mock_vector: AsyncMock,
        mock_lexical: AsyncMock,
    ) -> None:
        from app.retrieval.hybrid import hybrid_retrieve

        vector_hits = [_make_result("v1"), _make_result("shared")]
        lexical_hits = [_make_result("shared"), _make_result("l1")]
        mock_vector.return_value = vector_hits
        mock_lexical.return_value = lexical_hits

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])

        results = await hybrid_retrieve("test query", mock_embedder, top_n=10)

        mock_vector.assert_called_once()
        mock_lexical.assert_called_once()
        # "shared" appears in both lists → highest RRF score → first
        assert results[0].id == "shared"

    @patch("app.retrieval.hybrid.lexical_search")
    @patch("app.retrieval.hybrid.vector_search")
    async def test_result_count_capped_at_top_n(
        self,
        mock_vector: AsyncMock,
        mock_lexical: AsyncMock,
    ) -> None:
        from app.retrieval.hybrid import hybrid_retrieve

        mock_vector.return_value = [_make_result(f"v{i}") for i in range(20)]
        mock_lexical.return_value = [_make_result(f"l{i}") for i in range(20)]

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.0, 0.0, 0.0, 0.0]])

        results = await hybrid_retrieve("q", mock_embedder, top_n=5)
        assert len(results) <= 5


# ── Reranker tests (mocked Cohere) ──────────────────────────────────────────

class TestReranker:
    def _make_settings(self) -> Any:
        from app.config import Settings
        return Settings(
            cohere_api_key="test",
            database_url="postgresql://x:x@localhost/test",
            rerank_top_k=3,
        )

    def _make_candidates(self, n: int = 5) -> list[FusedResult]:
        return [
            FusedResult(
                id=f"doc{i}",
                source_key="src.txt",
                chunk_index=i,
                content=f"chunk {i}",
                rrf_score=1.0 / (i + 1),
            )
            for i in range(n)
        ]

    @patch("app.retrieval.reranker.cohere.AsyncClientV2")
    async def test_rerank_returns_top_k(self, mock_cohere_cls: MagicMock) -> None:
        from app.retrieval.reranker import Reranker

        mock_hit = MagicMock(index=0, relevance_score=0.95)
        mock_hit2 = MagicMock(index=2, relevance_score=0.80)
        mock_hit3 = MagicMock(index=1, relevance_score=0.60)

        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(
            return_value=MagicMock(results=[mock_hit, mock_hit2, mock_hit3])
        )
        mock_cohere_cls.return_value = mock_client

        settings = self._make_settings()
        reranker = Reranker(settings)
        candidates = self._make_candidates(5)

        results = await reranker.rerank("query", candidates)

        assert len(results) == 3
        # First result is the one with highest relevance score (index 0)
        assert results[0].relevance_score == pytest.approx(0.95)
        assert results[0].id == "doc0"

    @patch("app.retrieval.reranker.cohere.AsyncClientV2")
    async def test_rerank_empty_candidates_returns_empty(
        self, mock_cohere_cls: MagicMock
    ) -> None:
        from app.retrieval.reranker import Reranker

        mock_cohere_cls.return_value = AsyncMock()
        reranker = Reranker(self._make_settings())
        results = await reranker.rerank("query", [])
        assert results == []

    @patch("app.retrieval.reranker.cohere.AsyncClientV2")
    async def test_rerank_preserves_content(self, mock_cohere_cls: MagicMock) -> None:
        from app.retrieval.reranker import Reranker

        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(
            return_value=MagicMock(results=[MagicMock(index=1, relevance_score=0.9)])
        )
        mock_cohere_cls.return_value = mock_client

        reranker = Reranker(self._make_settings())
        candidates = self._make_candidates(3)
        results = await reranker.rerank("q", candidates)

        assert results[0].content == "chunk 1"
        assert results[0].source_key == "src.txt"

    @patch("app.retrieval.reranker.cohere.AsyncClientV2")
    async def test_rerank_top_k_override(self, mock_cohere_cls: MagicMock) -> None:
        from app.retrieval.reranker import Reranker

        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(
            return_value=MagicMock(
                results=[MagicMock(index=i, relevance_score=0.9 - i * 0.1) for i in range(2)]
            )
        )
        mock_cohere_cls.return_value = mock_client

        reranker = Reranker(self._make_settings())
        results = await reranker.rerank("q", self._make_candidates(5), top_k=2)

        # Cohere was asked for top_n=2
        call_kwargs = mock_client.rerank.call_args.kwargs
        assert call_kwargs["top_n"] == 2
        assert len(results) == 2
