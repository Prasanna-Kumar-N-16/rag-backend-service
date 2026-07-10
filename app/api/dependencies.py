"""FastAPI dependency providers for shared service singletons.

Service objects (Embedder, Reranker, ClaudeClient, Synthesizer) are created
once per process via module-level lazy initialization. Route handlers receive
them through ``Depends()`` so they are swappable in tests via
``app.dependency_overrides``.
"""

from __future__ import annotations

import threading

from fastapi import Depends

from app.config import Settings, get_settings
from app.generation.claude_client import ClaudeClient
from app.generation.synthesizer import Synthesizer
from app.indexing.embedder import Embedder
from app.retrieval.reranker import Reranker

# Module-level singletons — None until first request.
_embedder: Embedder | None = None
_reranker: Reranker | None = None
_claude_client: ClaudeClient | None = None
_synthesizer: Synthesizer | None = None

# Guards singleton construction. FastAPI runs sync dependencies in a
# threadpool, so concurrent first requests can otherwise race past the
# ``is None`` check and construct duplicate instances (leaking, for
# ClaudeClient/Reranker, an unused HTTP client).
_lock = threading.Lock()


def get_embedder(settings: Settings = Depends(get_settings)) -> Embedder:
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = Embedder(settings)
    return _embedder


def get_reranker(settings: Settings = Depends(get_settings)) -> Reranker:
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = Reranker(settings)
    return _reranker


def get_claude_client(settings: Settings = Depends(get_settings)) -> ClaudeClient:
    global _claude_client  # noqa: PLW0603
    if _claude_client is None:
        with _lock:
            if _claude_client is None:
                _claude_client = ClaudeClient(settings)
    return _claude_client


def get_synthesizer(
    client: ClaudeClient = Depends(get_claude_client),
    settings: Settings = Depends(get_settings),
) -> Synthesizer:
    global _synthesizer  # noqa: PLW0603
    if _synthesizer is None:
        with _lock:
            if _synthesizer is None:
                _synthesizer = Synthesizer(client=client, model=settings.generation_model)
    return _synthesizer


def reset_singletons() -> None:
    """Clear all cached singletons. Used in tests to reset state."""
    global _embedder, _reranker, _claude_client, _synthesizer  # noqa: PLW0603
    _embedder = None
    _reranker = None
    _claude_client = None
    _synthesizer = None
