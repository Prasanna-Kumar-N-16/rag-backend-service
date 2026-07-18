"""Application configuration.

All settings are environment-driven via ``pydantic-settings`` so that models,
provider keys, and tunables can be changed without code edits (12-factor).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "rag-backend-service"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # ── Database (PostgreSQL + pgvector) ─────────────────────────────────────
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10

    # ── Provider API keys (optional so the service can boot health-only) ─────
    anthropic_api_key: str | None = None
    voyage_api_key: str | None = None
    cohere_api_key: str | None = None

    # ── Inbound auth ─────────────────────────────────────────────────────────
    # Required to call /v1/query or /v1/ingest. None (unset) means those
    # routes are refused with a 500 rather than left open, so a forgotten
    # env var fails closed instead of exposing an unauthenticated service.
    api_key: str | None = None

    # ── Models ───────────────────────────────────────────────────────────────
    generation_model: str = "claude-opus-4-8"
    embedding_model: str = "voyage-3"
    rerank_model: str = "rerank-3"

    # ── Retrieval tunables ───────────────────────────────────────────────────
    retrieval_top_n: int = Field(default=40, ge=1)
    rerank_top_k: int = Field(default=6, ge=1)
    embedding_dimensions: int = Field(default=1024, ge=1)

    # ── Indexing tunables ────────────────────────────────────────────────────
    chunk_size: int = Field(default=400, ge=1)
    chunk_overlap: int = Field(default=80, ge=0)

    # ── AWS S3 (ingestion source) ────────────────────────────────────────────
    aws_region: str = "us-east-1"
    s3_bucket: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (one per process)."""
    return Settings()
