"""Tests for thread-safe singleton construction in app.api.dependencies."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from app.api.dependencies import get_embedder, get_indexer, reset_singletons
from app.config import Settings


def _settings() -> Settings:
    return Settings(database_url="postgresql://x:x@localhost/test")


class TestSingletonConcurrency:
    def test_concurrent_get_embedder_constructs_once(self) -> None:
        reset_singletons()
        construct_count = 0
        count_lock = threading.Lock()

        class FakeEmbedder:
            def __init__(self, settings: Settings) -> None:
                nonlocal construct_count
                # Widen the race window between the None-check and the
                # assignment so concurrent callers are likely to interleave.
                time.sleep(0.05)
                with count_lock:
                    construct_count += 1

        settings = _settings()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            get_embedder(settings)

        try:
            with patch("app.api.dependencies.Embedder", FakeEmbedder):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            assert construct_count == 1
        finally:
            reset_singletons()

    def test_concurrent_get_indexer_constructs_once(self) -> None:
        reset_singletons()
        construct_count = 0
        count_lock = threading.Lock()

        class FakeIndexer:
            def __init__(self, settings: Settings) -> None:
                nonlocal construct_count
                time.sleep(0.05)
                with count_lock:
                    construct_count += 1

        settings = _settings()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            get_indexer(settings)

        try:
            with patch("app.api.dependencies.Indexer", FakeIndexer):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            assert construct_count == 1
        finally:
            reset_singletons()

    def test_get_indexer_returns_same_instance_across_calls(self) -> None:
        reset_singletons()
        settings = _settings()
        try:
            with patch("app.api.dependencies.Indexer") as mock_indexer_cls:
                mock_indexer_cls.return_value = MagicMock()
                first = get_indexer(settings)
                second = get_indexer(settings)
                assert first is second
                mock_indexer_cls.assert_called_once()
        finally:
            reset_singletons()
