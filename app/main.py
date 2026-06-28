"""FastAPI application factory and entrypoint.

The ASGI app is exposed as ``app`` for ``uvicorn app.main:app``. A lifespan
context manager owns the asyncpg connection pool (opened on startup, closed on
shutdown) so connections are shared across the whole request lifetime.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import health
from app.config import Settings, get_settings
from app.db import close_db, init_db
from app.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources."""
    settings: Settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    logger = get_logger("app.startup")
    logger.info(
        "service_starting",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )

    await init_db(settings)

    try:
        yield
    finally:
        await close_db()
        get_logger("app.shutdown").info("service_stopping", service=settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        summary="Retrieval-Augmented Generation backend service.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    return app


app = create_app()
