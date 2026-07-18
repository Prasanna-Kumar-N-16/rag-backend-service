"""Smoke tests for the health/readiness endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "rag-backend-service"
    assert body["version"]


def test_readyz_ok(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
