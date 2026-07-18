"""Tests for the database bootstrap layer (no live Postgres required)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app import db


class TestInitConnection:
    async def test_creates_extension_before_registering_codec(self) -> None:
        """The vector extension must be created before the codec is registered.

        Regression guard: registering the pgvector codec before the extension
        exists fails with ``unknown type: public.vector`` on a fresh database.
        """
        conn = AsyncMock()
        calls: list[str] = []
        conn.execute.side_effect = lambda *a, **k: calls.append(f"execute:{a[0]}")

        async def fake_register(c: object) -> None:
            calls.append("register_vector")

        with patch.object(db, "register_vector", side_effect=fake_register):
            await db._init_connection(conn)

        assert calls == [
            "execute:CREATE EXTENSION IF NOT EXISTS vector",
            "register_vector",
        ]
