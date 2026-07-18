"""API-key authentication for inbound requests.

Applied at the router level to ``/v1/query`` and ``/v1/ingest`` — both are
cost-incurring (provider API calls) and data-exposing (answers synthesized
from private indexed documents), so neither should be reachable without a
shared secret. Health/readiness probes are intentionally excluded.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Raise if the request's ``X-API-Key`` header doesn't match configuration."""
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error.",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
