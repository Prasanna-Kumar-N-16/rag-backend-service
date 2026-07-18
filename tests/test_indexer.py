"""Tests for the indexing pipeline with mocked external dependencies.

All I/O (S3, Voyage AI, asyncpg) is mocked so the suite runs offline with no
API keys or live database.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings


def _make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "anthropic_api_key": "test",
        "voyage_api_key": "test",
        "cohere_api_key": "test",
        "s3_bucket": "test-bucket",
        "database_url": "postgresql://x:x@localhost/test",
        "embedding_dimensions": 4,  # tiny for testing
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# S3 loader tests
# ---------------------------------------------------------------------------

class TestS3Loader:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _make_settings()

    @patch("app.indexing.s3_loader.boto3.client")
    async def test_list_keys_paginates(
        self, mock_boto_client: MagicMock, settings: Settings
    ) -> None:
        from app.indexing.s3_loader import S3Loader

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "doc1.txt"}, {"Key": "doc2.txt"}]},
        ]
        mock_client.get_paginator.return_value = mock_paginator
        mock_boto_client.return_value = mock_client

        loader = S3Loader(settings)
        keys = await loader.list_keys(prefix="")
        assert keys == ["doc1.txt", "doc2.txt"]

    @patch("app.indexing.s3_loader.boto3.client")
    async def test_fetch_returns_s3_object(
        self, mock_boto_client: MagicMock, settings: Settings
    ) -> None:
        from app.indexing.s3_loader import S3Loader, S3Object

        mock_client = MagicMock()
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: b"hello content"),
            "ContentType": "text/plain",
        }
        mock_boto_client.return_value = mock_client

        loader = S3Loader(settings)
        obj = await loader.fetch("doc1.txt")
        assert isinstance(obj, S3Object)
        assert obj.content == b"hello content"
        assert obj.key == "doc1.txt"


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _make_settings()

    @patch("app.indexing.embedder.voyageai.AsyncClient")
    async def test_embed_returns_vectors(
        self, mock_voyage_cls: MagicMock, settings: Settings
    ) -> None:
        from app.indexing.embedder import Embedder

        fake_embedding = [0.1, 0.2, 0.3, 0.4]
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embeddings=[fake_embedding])
        mock_voyage_cls.return_value = mock_client

        embedder = Embedder(settings)
        result = await embedder.embed(["hello world"])
        assert result == [fake_embedding]

    @patch("app.indexing.embedder.voyageai.AsyncClient")
    async def test_embed_empty_list_returns_empty(
        self, mock_voyage_cls: MagicMock, settings: Settings
    ) -> None:
        from app.indexing.embedder import Embedder

        mock_voyage_cls.return_value = AsyncMock()
        embedder = Embedder(settings)
        result = await embedder.embed([])
        assert result == []


# ---------------------------------------------------------------------------
# Indexer tests
# ---------------------------------------------------------------------------

class TestIndexer:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _make_settings()

    @patch("app.indexing.indexer.get_pool")
    @patch("app.indexing.indexer.Embedder")
    @patch("app.indexing.indexer.S3Loader")
    async def test_index_key_upserts_chunks(
        self,
        mock_loader_cls: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_get_pool: MagicMock,
        settings: Settings,
    ) -> None:
        from app.indexing.indexer import Indexer

        # Set up S3 mock
        mock_loader = MagicMock()
        mock_loader.fetch = AsyncMock(
            return_value=MagicMock(
                key="doc.txt",
                content=b"This is a test document. " * 20,
            )
        )
        mock_loader_cls.return_value = mock_loader

        # Set up Voyage mock — return a vector per chunk
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(
            side_effect=lambda texts: [[0.1, 0.2, 0.3, 0.4]] * len(texts)
        )
        mock_embedder_cls.return_value = mock_embedder

        # Set up asyncpg mock
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_pool.return_value = mock_pool

        indexer = Indexer(settings)
        count = await indexer.index_key("doc.txt")

        assert count >= 1
        assert mock_conn.executemany.called
        # All chunk rows go out in a single batched executemany call, never as
        # concurrent per-chunk executes on the shared connection.
        (_sql, rows), _kwargs = mock_conn.executemany.call_args
        assert len(rows) == count

    @patch("app.indexing.indexer.get_pool")
    @patch("app.indexing.indexer.Embedder")
    @patch("app.indexing.indexer.S3Loader")
    async def test_content_hash_is_sha256(
        self,
        mock_loader_cls: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_get_pool: MagicMock,
        settings: Settings,
    ) -> None:
        """Verify the content_hash passed to the DB is SHA-256 of chunk text."""
        from app.indexing.indexer import _content_hash

        text = "short doc"
        assert _content_hash(text) == hashlib.sha256(text.encode()).hexdigest()

    @patch("app.indexing.indexer.get_pool")
    @patch("app.indexing.indexer.Embedder")
    @patch("app.indexing.indexer.S3Loader")
    async def test_index_prefix_aggregates_counts(
        self,
        mock_loader_cls: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_get_pool: MagicMock,
        settings: Settings,
    ) -> None:
        from app.indexing.indexer import Indexer
        from app.indexing.s3_loader import S3Object

        objects = [
            S3Object(key="a.txt", content=b"doc a " * 30, content_type="text/plain"),
            S3Object(key="b.txt", content=b"doc b " * 30, content_type="text/plain"),
        ]

        async def _fake_iter(prefix: str = ""):  # type: ignore[no-untyped-def]
            for obj in objects:
                yield obj

        mock_loader = MagicMock()
        mock_loader.iter_prefix = _fake_iter
        mock_loader_cls.return_value = mock_loader

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(
            side_effect=lambda texts: [[0.0, 0.0, 0.0, 0.0]] * len(texts)
        )
        mock_embedder_cls.return_value = mock_embedder

        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock(return_value=None)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_pool.return_value = mock_pool

        indexer = Indexer(settings)
        result = await indexer.index_prefix("")
        assert result["processed"] == 2
        assert result["chunks"] >= 2

    @patch("app.indexing.indexer.get_pool")
    @patch("app.indexing.indexer.Embedder")
    @patch("app.indexing.indexer.S3Loader")
    async def test_upsert_never_runs_concurrent_ops_on_one_connection(
        self,
        mock_loader_cls: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_get_pool: MagicMock,
        settings: Settings,
    ) -> None:
        """Regression guard for the asyncpg single-operation-per-connection rule.

        A real asyncpg connection raises ``InterfaceError`` if a second
        operation starts while another is in flight. This fake connection
        enforces the same invariant, so reintroducing a parallel
        ``asyncio.gather(conn.execute(...))`` pattern would fail this test.
        """
        import asyncio

        from app.indexing.indexer import Indexer

        mock_loader = MagicMock()
        mock_loader.fetch = AsyncMock(
            return_value=MagicMock(
                key="doc.txt",
                content=b"This is a test document. " * 200,
            )
        )
        mock_loader_cls.return_value = mock_loader

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(
            side_effect=lambda texts: [[0.1, 0.2, 0.3, 0.4]] * len(texts)
        )
        mock_embedder_cls.return_value = mock_embedder

        class SingleOpConnection:
            """Rejects overlapping operations the way asyncpg does."""

            def __init__(self) -> None:
                self._busy = False
                self.executemany_calls = 0

            async def _run(self) -> None:
                if self._busy:
                    raise RuntimeError(
                        "cannot perform operation: another operation is in progress"
                    )
                self._busy = True
                await asyncio.sleep(0)  # yield so overlapping callers collide
                self._busy = False

            async def execute(self, *args: Any, **kwargs: Any) -> None:
                await self._run()

            async def executemany(self, *args: Any, **kwargs: Any) -> None:
                self.executemany_calls += 1
                await self._run()

        conn = SingleOpConnection()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_pool.return_value = mock_pool

        indexer = Indexer(settings)
        count = await indexer.index_key("doc.txt")

        assert count > 1  # multi-chunk doc, so the bug would have surfaced
        assert conn.executemany_calls == 1
