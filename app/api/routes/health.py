"""Liveness and readiness endpoints.

``/healthz`` is a pure liveness probe (process is up). ``/readyz`` is a
readiness probe that, from Day 2 onward, will also verify the database pool.
For Day 1 it reports ready unconditionally.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.config import get_settings

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
    """Return ``ok`` when the service is ready to accept traffic.

    Day 1: always ready. A later iteration adds a database-pool check here.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
