"""Database layer: asyncpg connection pool + pgvector schema bootstrap.

Usage
-----
Call ``init_db(settings)`` during the FastAPI lifespan to open the pool and
run the DDL migrations. Call ``close_db()`` on shutdown. Use ``get_pool()``
anywhere you need a connection.
"""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import Settings
from app.logging import get_logger

logger = get_logger(__name__)

_pool: asyncpg.Pool | None = None

# DDL executed once at startup (idempotent via IF NOT EXISTS).
_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id            TEXT        PRIMARY KEY,
    source_key    TEXT        NOT NULL,
    chunk_index   INTEGER     NOT NULL,
    content       TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,
    embedding     vector({dims}),
    tsv           tsvector    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_key, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
    ON document_chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS document_chunks_hash_idx
    ON document_chunks (content_hash);
"""


async def init_db(settings: Settings) -> None:
    """Open the asyncpg pool and bootstrap the pgvector schema."""
    global _pool  # noqa: PLW0603

    logger.info("db_connecting", dsn=settings.database_url.split("@")[-1])
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        # Register the pgvector codec for every new connection in the pool.
        init=register_vector,
    )

    async with _pool.acquire() as conn:
        await conn.execute(
            _SCHEMA_SQL.format(dims=settings.embedding_dimensions)
        )

    logger.info("db_ready", pool_min=settings.db_pool_min_size, pool_max=settings.db_pool_max_size)


async def close_db() -> None:
    """Close the asyncpg connection pool gracefully."""
    global _pool  # noqa: PLW0603
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db_closed")


def get_pool() -> asyncpg.Pool:
    """Return the active connection pool.

    Raises:
        RuntimeError: If called before ``init_db`` has completed.
    """
    if _pool is None:
        raise RuntimeError("Database pool is not initialised. Call init_db() first.")
    return _pool


async def pool_is_healthy() -> bool:
    """Quick liveness check: acquire a connection and run SELECT 1."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False
