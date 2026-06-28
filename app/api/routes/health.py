"""Liveness and readiness endpoints.

``/healthz`` is a pure liveness probe (process is up).
``/readyz`` additionally verifies the database pool is reachable.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.config import get_settings
from app.db import pool_is_healthy

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health/readiness payload."""

    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    """Return ``ok`` if the process is alive."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
async def readyz() -> HealthResponse:
    """Return ``ok`` when the service is ready (DB pool reachable)."""
    settings = get_settings()
    db_ok = await pool_is_healthy()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
