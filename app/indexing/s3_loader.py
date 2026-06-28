"""Fetch document objects from AWS S3.

boto3 is synchronous, so all calls are offloaded to a thread pool via
``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class S3Object:
    """Represents a single S3 object retrieved for indexing."""

    key: str
    content: bytes
    content_type: str


def _list_keys(bucket: str, prefix: str, client: Any) -> list[str]:
    """List all object keys under *prefix* in *bucket* (handles pagination)."""
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _get_object(bucket: str, key: str, client: Any) -> S3Object:
    resp = client.get_object(Bucket=bucket, Key=key)
    return S3Object(
        key=key,
        content=resp["Body"].read(),
        content_type=resp.get("ContentType", "application/octet-stream"),
    )


class S3Loader:
    """Asynchronous S3 document loader."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.s3_bucket or ""
        self._client: Any = boto3.client("s3", region_name=settings.aws_region)

    async def list_keys(self, prefix: str = "") -> list[str]:
        """Return all object keys under *prefix* (async, non-blocking)."""
        return await asyncio.to_thread(_list_keys, self._bucket, prefix, self._client)

    async def fetch(self, key: str) -> S3Object:
        """Download a single S3 object (async, non-blocking)."""
        try:
            return await asyncio.to_thread(_get_object, self._bucket, key, self._client)
        except (BotoCoreError, ClientError) as exc:
            logger.error("s3_fetch_failed", key=key, error=str(exc))
            raise

    async def iter_prefix(self, prefix: str = "") -> AsyncIterator[S3Object]:
        """Yield all objects under *prefix* one by one."""
        keys = await self.list_keys(prefix)
        logger.info("s3_keys_found", bucket=self._bucket, prefix=prefix, count=len(keys))
        for key in keys:
            yield await self.fetch(key)
