"""Orchestrate the ingestion pipeline: S3 → chunk → embed → upsert.

Idempotency
-----------
Each chunk is identified by a SHA-256 of its text content. If a row with the
same ``id`` (source_key + chunk_index) already exists *and* the
``content_hash`` matches, the upsert is a no-op — reindexing the same document
twice is safe and cheap.
"""

from __future__ import annotations

import hashlib

import asyncpg

from app.config import Settings
from app.db import get_pool
from app.indexing.chunker import chunk_text
from app.indexing.embedder import Embedder
from app.indexing.s3_loader import S3Loader
from app.logging import get_logger

logger = get_logger(__name__)

_UPSERT_SQL = """
INSERT INTO document_chunks
    (id, source_key, chunk_index, content, content_hash, embedding)
VALUES
    ($1, $2, $3, $4, $5, $6)
ON CONFLICT (id) DO UPDATE
    SET content      = EXCLUDED.content,
        content_hash = EXCLUDED.content_hash,
        embedding    = EXCLUDED.embedding,
        created_at   = now()
    WHERE document_chunks.content_hash != EXCLUDED.content_hash
"""


def _chunk_id(source_key: str, chunk_index: int) -> str:
    return f"{source_key}#{chunk_index}"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Indexer:
    """End-to-end document indexer."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loader = S3Loader(settings)
        self._embedder = Embedder(settings)

    async def index_prefix(self, s3_prefix: str = "") -> dict[str, int]:
        """Index all S3 objects under *s3_prefix*.

        Returns:
            A dict with ``"processed"`` and ``"chunks"`` counts.
        """
        processed = 0
        total_chunks = 0

        async for s3_obj in self._loader.iter_prefix(s3_prefix):
            try:
                text = s3_obj.content.decode("utf-8", errors="replace")
                chunks_indexed = await self._index_document(s3_obj.key, text)
                total_chunks += chunks_indexed
                processed += 1
                logger.info(
                    "document_indexed",
                    key=s3_obj.key,
                    chunks=chunks_indexed,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("document_index_failed", key=s3_obj.key, error=str(exc))

        logger.info("prefix_indexed", prefix=s3_prefix, processed=processed, chunks=total_chunks)
        return {"processed": processed, "chunks": total_chunks}

    async def index_key(self, s3_key: str) -> int:
        """Index a single S3 object by *s3_key*. Returns chunk count."""
        obj = await self._loader.fetch(s3_key)
        text = obj.content.decode("utf-8", errors="replace")
        return await self._index_document(s3_key, text)

    async def _index_document(self, source_key: str, text: str) -> int:
        """Chunk, embed, and upsert all chunks for one document."""
        chunks = chunk_text(
            text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = await self._embedder.embed(texts)

        # asyncpg forbids concurrent operations on a single connection, so the
        # per-chunk upserts must not be fired off in parallel on one ``conn``.
        # ``executemany`` runs them sequentially over the same connection in a
        # single round-trip batch — correct and faster than gathering executes.
        rows = [
            (
                _chunk_id(source_key, chunk.index),
                source_key,
                chunk.index,
                chunk.text,
                _content_hash(chunk.text),
                embeddings[i],
            )
            for i, chunk in enumerate(chunks)
        ]

        pool: asyncpg.Pool = get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(_UPSERT_SQL, rows)

        return len(chunks)
