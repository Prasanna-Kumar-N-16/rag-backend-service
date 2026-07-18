"""POST /v1/ingest — trigger S3 document indexing.

The indexing job is launched as a FastAPI ``BackgroundTask`` so the endpoint
returns immediately with a 202 Accepted while the pipeline runs asynchronously.

Production graduation note: replace BackgroundTasks with a proper worker queue
(Celery / Arq / SQS) for reliability, retries, and observability at scale.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.api.dependencies import get_indexer
from app.api.schemas import IngestRequest, IngestResponse
from app.api.security import require_api_key
from app.config import Settings, get_settings
from app.indexing.indexer import Indexer
from app.logging import get_logger

router = APIRouter(prefix="/v1", tags=["ingest"], dependencies=[Depends(require_api_key)])
logger = get_logger(__name__)


async def _run_indexing(indexer: Indexer, s3_prefix: str) -> None:
    """Background task: run the pipeline on the shared indexer singleton."""
    try:
        result = await indexer.index_prefix(s3_prefix)
        logger.info("ingest_complete", prefix=s3_prefix, **result)
    except Exception as exc:  # noqa: BLE001
        logger.error("ingest_failed", prefix=s3_prefix, error=str(exc))


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger S3 ingestion",
)
async def ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Enqueue an S3 prefix for chunking, embedding, and indexing."""
    if not settings.s3_bucket:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="S3_BUCKET is not configured on this server.",
        )
    if not settings.voyage_api_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="VOYAGE_API_KEY is not configured on this server.",
        )

    # Resolved after the checks above (not as a route-level Depends) so a
    # missing S3_BUCKET/VOYAGE_API_KEY still yields a clean 422 instead of
    # failing during Indexer construction.
    indexer = get_indexer(settings)

    logger.info("ingest_queued", prefix=body.s3_prefix, bucket=settings.s3_bucket)
    background_tasks.add_task(_run_indexing, indexer, body.s3_prefix)

    return IngestResponse(
        status="accepted",
        message=f"Indexing job queued for s3://{settings.s3_bucket}/{body.s3_prefix}",
    )
