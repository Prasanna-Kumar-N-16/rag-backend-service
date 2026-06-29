"""Pydantic request and response schemas for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── /v1/query ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /v1/query request body."""

    query: str = Field(..., min_length=1, max_length=4096, description="User question.")
    top_n: int = Field(default=40, ge=1, le=200, description="Candidate retrieval count per leg.")
    top_k: int = Field(default=6, ge=1, le=40, description="Final chunks passed to generation.")
    max_tokens: int = Field(default=2048, ge=64, le=8192, description="Max answer tokens.")


class SourceSchema(BaseModel):
    """A single cited source chunk."""

    citation_number: int
    source_key: str
    chunk_index: int
    content_preview: str


class QueryResponse(BaseModel):
    """POST /v1/query response body."""

    answer: str
    sources: list[SourceSchema]
    model: str


# ── /v1/ingest ───────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """POST /v1/ingest request body."""

    s3_prefix: str = Field(
        default="",
        description="S3 key prefix to index. Empty string indexes the entire bucket.",
    )


class IngestResponse(BaseModel):
    """POST /v1/ingest response body."""

    status: str
    message: str
