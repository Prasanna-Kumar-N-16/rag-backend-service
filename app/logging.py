"""Structured logging setup via ``structlog``.

Call :func:`configure_logging` once at startup. Emits JSON in production-like
environments and a colored console renderer otherwise.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog


def configure_logging(*, level: str = "INFO", json_logs: bool = True) -> None:
    """Configure the stdlib root logger and ``structlog`` processors.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        json_logs: Emit JSON lines when ``True``; otherwise a console renderer.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound ``structlog`` logger."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
