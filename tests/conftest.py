"""Shared test fixtures.

The FastAPI lifespan now opens a database pool, so tests that don't need a
real DB (all Day 1/2 tests) patch ``init_db`` and ``close_db`` to no-ops and
stub ``pool_is_healthy`` to return True.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Yield a TestClient with the DB pool mocked out."""
    with (
        patch("app.main.init_db", new=AsyncMock()),
        patch("app.main.close_db", new=AsyncMock()),
        patch("app.api.routes.health.pool_is_healthy", new=AsyncMock(return_value=True)),
        TestClient(create_app()) as test_client,
    ):
        yield test_client
