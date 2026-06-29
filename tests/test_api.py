"""End-to-end API tests for /v1/query and /v1/ingest.

External I/O is mocked via FastAPI's ``app.dependency_overrides`` and
``unittest.mock.patch``. The suite runs fully offline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_embedder, get_reranker, get_synthesizer
from app.generation.synthesizer import Source, SynthesisResult
from app.retrieval.hybrid import FusedResult
from app.retrieval.reranker import RankedResult

# ── shared mock factories ────────────────────────────────────────────────────

def _make_fused(n: int = 5) -> list[FusedResult]:
    return [
        FusedResult(
            id=f"d{i}", source_key=f"doc{i}.txt", chunk_index=i,
            content=f"chunk {i}", rrf_score=1.0 / (i + 1),
        )
        for i in range(n)
    ]


def _make_ranked(n: int = 3) -> list[RankedResult]:
    return [
        RankedResult(
            id=f"d{i}", source_key=f"doc{i}.txt", chunk_index=i,
            content=f"chunk {i}", relevance_score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


def _make_synthesis(answer: str = "The answer is 42.") -> SynthesisResult:
    return SynthesisResult(
        answer=answer,
        sources=[
            Source(citation_number=1, source_key="doc0.txt",
                   chunk_index=0, content_preview="chunk 0"),
        ],
        model="claude-opus-4-8",
    )


def _mock_pipeline(
    app: FastAPI,
    fused: list[FusedResult] | None = None,
    ranked: list[RankedResult] | None = None,
    synthesis: SynthesisResult | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Override all service deps so tests run without any API keys.

    FastAPI resolves ALL declared ``Depends()`` before the route handler runs,
    so even when ``hybrid_retrieve`` is patched, the embedder/reranker/
    synthesizer dependencies still get resolved. We must override all of them.
    """
    mock_embedder = MagicMock()  # hybrid_retrieve is patched, so this is never called
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=ranked or _make_ranked())
    mock_synth = MagicMock()
    mock_synth.synthesize = AsyncMock(return_value=synthesis or _make_synthesis())

    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_synthesizer] = lambda: mock_synth

    return mock_reranker, mock_synth


# ── /v1/query tests ──────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_happy_path_returns_200(self, app: FastAPI, client: TestClient) -> None:
        _mock_pipeline(app)
        with patch(
            "app.api.routes.query.hybrid_retrieve",
            new=AsyncMock(return_value=_make_fused()),
        ):
            resp = client.post("/v1/query", json={"query": "What is the meaning of life?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "The answer is 42."
        assert len(body["sources"]) == 1
        assert body["model"] == "claude-opus-4-8"

    def test_response_schema_fields_present(self, app: FastAPI, client: TestClient) -> None:
        _mock_pipeline(app)
        with patch(
            "app.api.routes.query.hybrid_retrieve",
            new=AsyncMock(return_value=_make_fused()),
        ):
            resp = client.post("/v1/query", json={"query": "test"})

        body = resp.json()
        for field in ("answer", "sources", "model"):
            assert field in body
        source = body["sources"][0]
        for field in ("citation_number", "source_key", "chunk_index", "content_preview"):
            assert field in source

    def test_empty_query_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={"top_k": 3})
        assert resp.status_code == 422

    def test_custom_request_id_echoed(self, app: FastAPI, client: TestClient) -> None:
        _mock_pipeline(app)
        with patch(
            "app.api.routes.query.hybrid_retrieve",
            new=AsyncMock(return_value=_make_fused()),
        ):
            resp = client.post(
                "/v1/query",
                json={"query": "test"},
                headers={"X-Request-ID": "my-req-123"},
            )
        assert resp.headers.get("X-Request-ID") == "my-req-123"

    def test_auto_request_id_generated(self, app: FastAPI, client: TestClient) -> None:
        _mock_pipeline(app)
        with patch(
            "app.api.routes.query.hybrid_retrieve",
            new=AsyncMock(return_value=_make_fused()),
        ):
            resp = client.post("/v1/query", json={"query": "test"})
        assert resp.headers.get("X-Request-ID")  # truthy UUID

    def test_rate_limit_error_returns_503(self, app: FastAPI, client: TestClient) -> None:
        import anthropic

        # Still need embedder/reranker/synthesizer overrides — FastAPI resolves
        # all Depends() before the route body runs (and hybrid_retrieve raises
        # inside the route body, not during dependency resolution).
        _mock_pipeline(app)
        with patch(
            "app.api.routes.query.hybrid_retrieve",
            new=AsyncMock(
                side_effect=anthropic.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body={},
                )
            ),
        ):
            resp = client.post("/v1/query", json={"query": "test"})
        assert resp.status_code == 503

    def test_no_context_returns_200_with_empty_sources(
        self, app: FastAPI, client: TestClient
    ) -> None:
        _mock_pipeline(
            app,
            ranked=[],
            synthesis=SynthesisResult(
                answer="I could not find relevant information.",
                sources=[],
                model="claude-opus-4-8",
            ),
        )
        with patch(
            "app.api.routes.query.hybrid_retrieve", new=AsyncMock(return_value=[])
        ):
            resp = client.post("/v1/query", json={"query": "unknown topic"})
        assert resp.status_code == 200
        assert resp.json()["sources"] == []

    def test_top_k_param_forwarded(self, app: FastAPI, client: TestClient) -> None:
        mock_reranker, _ = _mock_pipeline(app)
        with patch(
            "app.api.routes.query.hybrid_retrieve",
            new=AsyncMock(return_value=_make_fused()),
        ):
            client.post("/v1/query", json={"query": "test", "top_k": 2})
        mock_reranker.rerank.assert_called_once()
        _, kwargs = mock_reranker.rerank.call_args
        assert kwargs.get("top_k") == 2 or mock_reranker.rerank.call_args.args[2] == 2


# ── /v1/ingest tests ─────────────────────────────────────────────────────────

class TestIngestEndpoint:
    def test_ingest_returns_202(self, client: TestClient) -> None:
        with patch("app.api.routes.ingest._run_indexing", new=AsyncMock()):
            resp = client.post("/v1/ingest", json={"s3_prefix": "docs/"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "accepted"
        assert "docs/" in body["message"]

    def test_ingest_default_prefix(self, client: TestClient) -> None:
        with patch("app.api.routes.ingest._run_indexing", new=AsyncMock()):
            resp = client.post("/v1/ingest", json={})
        assert resp.status_code == 202

    def test_ingest_no_bucket_returns_422(self, app: FastAPI, client: TestClient) -> None:
        from app.config import Settings
        from app.config import get_settings as _get_settings

        no_bucket = Settings(
            s3_bucket=None,
            voyage_api_key="test",
            database_url="postgresql://x:x@localhost/test",
        )
        app.dependency_overrides[_get_settings] = lambda: no_bucket
        resp = client.post("/v1/ingest", json={"s3_prefix": "docs/"})
        assert resp.status_code == 422

    def test_ingest_no_voyage_key_returns_422(self, app: FastAPI, client: TestClient) -> None:
        from app.config import Settings
        from app.config import get_settings as _get_settings

        no_key = Settings(
            s3_bucket="my-bucket",
            voyage_api_key=None,
            database_url="postgresql://x:x@localhost/test",
        )
        app.dependency_overrides[_get_settings] = lambda: no_key
        resp = client.post("/v1/ingest", json={"s3_prefix": "docs/"})
        assert resp.status_code == 422


# ── health endpoints still pass with new middleware ──────────────────────────

class TestHealthWithNewRoutes:
    def test_healthz_200(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200

    def test_readyz_200(self, client: TestClient) -> None:
        assert client.get("/readyz").status_code == 200

    def test_healthz_has_request_id(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.headers.get("X-Request-ID")


# ── claude_client unit tests ─────────────────────────────────────────────────

class TestClaudeClient:
    def _make_settings(self) -> MagicMock:
        s = MagicMock()
        s.anthropic_api_key = "test-key"
        s.generation_model = "claude-opus-4-8"
        return s

    @patch("app.generation.claude_client.anthropic.AsyncAnthropic")
    async def test_generate_returns_text(self, mock_anthropic_cls: MagicMock) -> None:
        from app.generation.claude_client import ClaudeClient

        mock_text_block = MagicMock(type="text", text="Hello from Claude")
        mock_message = MagicMock(
            content=[mock_text_block],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=5),
        )
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.get_final_message = AsyncMock(return_value=mock_message)

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        mock_anthropic_cls.return_value = mock_client

        cc = ClaudeClient(self._make_settings())
        result = await cc.generate("system", "user question")
        assert result == "Hello from Claude"

    @patch("app.generation.claude_client.anthropic.AsyncAnthropic")
    async def test_generate_no_text_block_returns_empty(
        self, mock_anthropic_cls: MagicMock
    ) -> None:
        from app.generation.claude_client import ClaudeClient

        mock_message = MagicMock(
            content=[MagicMock(type="tool_use")],
            stop_reason="tool_use",
            usage=MagicMock(input_tokens=5, output_tokens=0),
        )
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.get_final_message = AsyncMock(return_value=mock_message)

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        mock_anthropic_cls.return_value = mock_client

        cc = ClaudeClient(self._make_settings())
        result = await cc.generate("sys", "q")
        assert result == ""


# ── synthesizer unit tests ────────────────────────────────────────────────────

class TestSynthesizer:
    def _make_ranked(self, n: int = 2) -> list[RankedResult]:
        return [
            RankedResult(
                id=f"d{i}", source_key=f"src{i}.txt", chunk_index=i,
                content=f"relevant content {i}", relevance_score=0.9 - i * 0.1,
            )
            for i in range(n)
        ]

    async def test_synthesize_returns_result(self) -> None:
        from app.generation.synthesizer import Synthesizer

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="Answer with [1] citation.")
        synth = Synthesizer(client=mock_client, model="claude-opus-4-8")

        result = await synth.synthesize("what?", self._make_ranked())
        assert result.answer == "Answer with [1] citation."
        assert len(result.sources) == 2
        assert result.sources[0].citation_number == 1
        assert result.model == "claude-opus-4-8"

    async def test_synthesize_empty_ranked_skips_llm(self) -> None:
        from app.generation.synthesizer import Synthesizer

        mock_client = AsyncMock()
        synth = Synthesizer(client=mock_client, model="claude-opus-4-8")

        result = await synth.synthesize("what?", [])
        mock_client.generate.assert_not_called()
        assert result.sources == []
        assert "could not find" in result.answer.lower()

    async def test_synthesize_context_contains_numbered_passages(self) -> None:
        from app.generation.synthesizer import Synthesizer

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="ok")
        synth = Synthesizer(client=mock_client, model="claude-opus-4-8")

        await synth.synthesize("q", self._make_ranked(3))
        call_args = mock_client.generate.call_args
        user_prompt: str = call_args.kwargs.get("user_prompt") or call_args.args[1]
        assert "[1]" in user_prompt
        assert "[2]" in user_prompt
        assert "[3]" in user_prompt

    async def test_source_preview_truncated_to_200_chars(self) -> None:
        from app.generation.synthesizer import Synthesizer

        long_content = "x" * 500
        ranked = [
            RankedResult(
                id="d0", source_key="src.txt", chunk_index=0,
                content=long_content, relevance_score=0.9,
            )
        ]
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="ans")
        synth = Synthesizer(client=mock_client, model="m")

        result = await synth.synthesize("q", ranked)
        assert len(result.sources[0].content_preview) == 200


class TestQueryTopKForwarding:
    """Verify top_k is correctly passed through the query route."""

    def test_top_k_default_is_6(self, app: FastAPI, client: TestClient) -> None:
        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock(return_value=_make_ranked())
        mock_synth = MagicMock()
        mock_synth.synthesize = AsyncMock(return_value=_make_synthesis())
        app.dependency_overrides[get_embedder] = lambda: MagicMock()
        app.dependency_overrides[get_reranker] = lambda: mock_reranker
        app.dependency_overrides[get_synthesizer] = lambda: mock_synth

        with patch(
            "app.api.routes.query.hybrid_retrieve",
            new=AsyncMock(return_value=_make_fused()),
        ):
            resp = client.post("/v1/query", json={"query": "hi"})

        assert resp.status_code == 200
        mock_reranker.rerank.assert_called_once()
        # top_k=6 is the default from QueryRequest
        call = mock_reranker.rerank.call_args
        top_k_passed = call.kwargs.get("top_k") or (call.args[2] if len(call.args) > 2 else None)
        assert top_k_passed == 6


@pytest.mark.parametrize("top_k,expected", [(1, 1), (4, 4), (10, 10)])
def test_query_top_k_validation(
    top_k: int, expected: int, app: FastAPI, client: TestClient
) -> None:
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=_make_ranked(min(expected, 3)))
    mock_synth = MagicMock()
    mock_synth.synthesize = AsyncMock(return_value=_make_synthesis())
    app.dependency_overrides[get_embedder] = lambda: MagicMock()
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_synthesizer] = lambda: mock_synth

    with patch(
        "app.api.routes.query.hybrid_retrieve",
        new=AsyncMock(return_value=_make_fused()),
    ):
        resp = client.post("/v1/query", json={"query": "hi", "top_k": top_k})

    assert resp.status_code == 200
    call = mock_reranker.rerank.call_args
    top_k_passed = call.kwargs.get("top_k") or (call.args[2] if len(call.args) > 2 else None)
    assert top_k_passed == expected
