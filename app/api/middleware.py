"""Request-scoped middleware: request ID injection and response timing.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) to avoid
the known Starlette/anyio ExceptionGroup incompatibility in Python 3.12 where
``BaseHTTPMiddleware`` wraps responses in a TaskGroup that cannot cleanly
propagate FastAPI 422 validation errors.
"""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging import get_logger

logger = get_logger(__name__)

# Client-supplied request IDs are trusted into structured logs and echoed
# back in the response header, so only accept a conservative, bounded
# charset instead of arbitrary client-controlled bytes (log injection/flood).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIDMiddleware:
    """Attach ``X-Request-ID`` to every request/response and log timing."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        # Extract or generate request ID from raw headers. Client-supplied
        # values are only trusted if they match the safe charset above —
        # otherwise a fresh UUID is generated.
        raw_headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        client_request_id = next(
            (
                v.decode("utf-8", errors="replace")
                for k, v in raw_headers
                if k.lower() == b"x-request-id"
            ),
            None,
        )
        request_id = (
            client_request_id
            if client_request_id and _REQUEST_ID_RE.match(client_request_id)
            else str(uuid.uuid4())
        )

        start = time.perf_counter()

        async def send_with_extras(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                await self._app(scope, receive, send_with_extras)
            finally:
                if scope["type"] == "http":
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    path = scope.get("path", "")
                    method = scope.get("method", "")
                    logger.info(
                        "request_completed",
                        method=method,
                        path=path,
                        duration_ms=duration_ms,
                        request_id=request_id,
                    )
