"""Voyage AI embedding wrapper with batched inference.

Uses ``voyageai.AsyncClient`` for non-blocking calls. Batches are split to
respect the Voyage API's per-request input limit (128 texts / request).
"""

from __future__ import annotations

import asyncio
from typing import Any

import voyageai

from app.config import Settings
from app.logging import get_logger

logger = get_logger(__name__)

_VOYAGE_BATCH_SIZE = 128


class Embedder:
    """Async Voyage AI text embedder."""

    def __init__(self, settings: Settings) -> None:
        if not settings.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY is required for embedding.")
        self._model = settings.embedding_model
        self._dims = settings.embedding_dimensions
        self._client: Any = voyageai.AsyncClient(api_key=settings.voyage_api_key)  # type: ignore[attr-defined]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches and return a flat list of float vectors.

        Args:
            texts: Plain-text strings to embed.

        Returns:
            List of embedding vectors, one per input text, in the same order.
        """
        if not texts:
            return []

        batches = [
            texts[i : i + _VOYAGE_BATCH_SIZE]
            for i in range(0, len(texts), _VOYAGE_BATCH_SIZE)
        ]

        tasks = [self._embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        all_embeddings: list[list[float]] = []
        for batch_result in results:
            all_embeddings.extend(batch_result)

        logger.debug("embedded", count=len(texts), batches=len(batches))
        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self._model, input_type="document")
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in result.embeddings]
