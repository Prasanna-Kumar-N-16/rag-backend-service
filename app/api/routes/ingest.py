"""POST /v1/ingest — trigger S3 document indexing.

The indexing job is launched as a FastAPI ``BackgroundTask`` so the endpoint
returns immediately with a 202 Accepted while the pipeline runs asynchronously.

Production graduation note: replace BackgroundTasks with a proper worker queue
(Celery / Arq / SQS) for reliability, retries, and observability at scale.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.api.schemas import IngestRequest, IngestResponse
from app.config import Settings, get_settings
from app.indexing.indexer import Indexer
from app.logging import get_logger

router = APIRouter(prefix="/v1", tags=["ingest"])
logger = get_logger(__name__)


async def _run_indexing(settings: Settings, s3_prefix: str) -> None:
    """Background task: instantiate indexer and run the pipeline."""
    try:
        indexer = Indexer(settings)
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

    logger.info("ingest_queued", prefix=body.s3_prefix, bucket=settings.s3_bucket)
    background_tasks.add_task(_run_indexing, settings, body.s3_prefix)

    return IngestResponse(
        status="accepted",
        message=f"Indexing job queued for s3://{settings.s3_bucket}/{body.s3_prefix}",
    )
