"""Shared test fixtures.

Day 1 provides a FastAPI ``TestClient``. Later days add mocked Anthropic /
Voyage / Cohere clients and a fake database pool here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Yield a TestClient bound to a freshly built app (runs lifespan)."""
    with TestClient(create_app()) as test_client:
        yield test_client
