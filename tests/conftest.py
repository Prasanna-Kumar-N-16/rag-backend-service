"""Shared test fixtures.

The ``app`` fixture yields a FastAPI instance with:
  * DB lifecycle mocked (no live Postgres)
  * All service singletons (Embedder, Reranker, Synthesizer, ClaudeClient)
    pre-stubbed so that API keys are never required in tests
  * Overridable test settings with dummy values for s3_bucket, API keys, etc.

Individual tests may replace any of these overrides via
``app.dependency_overrides[dep] = lambda: custom_mock``.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_claude_client,
    get_embedder,
    get_reranker,
    get_synthesizer,
)
from app.config import Settings, get_settings
from app.main import create_app

# Test settings with all required fields populated so routes never raise
# "not configured" HTTPExceptions during normal test setup.
_TEST_SETTINGS = Settings(
    anthropic_api_key="test-anthropic",
    voyage_api_key="test-voyage",
    cohere_api_key="test-cohere",
    s3_bucket="test-bucket",
    database_url="postgresql://x:x@localhost/test",
    environment="development",
    log_json=False,
)


@pytest.fixture()
def app() -> Iterator[FastAPI]:
    """Yield a fully-mocked FastAPI app — no API keys or Postgres needed."""
    with (
        patch("app.main.init_db", new=AsyncMock()),
        patch("app.main.close_db", new=AsyncMock()),
        patch("app.api.routes.health.pool_is_healthy", new=AsyncMock(return_value=True)),
    ):
        application = create_app()

        # Provide test settings with all required keys filled in.
        application.dependency_overrides[get_settings] = lambda: _TEST_SETTINGS

        # Stub all service singletons; individual tests override as needed.
        application.dependency_overrides[get_embedder] = lambda: MagicMock()
        application.dependency_overrides[get_reranker] = lambda: MagicMock()
        application.dependency_overrides[get_claude_client] = lambda: MagicMock()
        application.dependency_overrides[get_synthesizer] = lambda: MagicMock()

        yield application

        application.dependency_overrides.clear()


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    """Yield a TestClient bound to the mocked app."""
    with TestClient(app) as test_client:
        yield test_client
