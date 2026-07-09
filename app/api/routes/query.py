"""POST /v1/query — full RAG pipeline with Responsible-AI guardrails.

Pipeline per request:
  [Guard-in]  1. Prompt-injection check on user query
  [Guard-in]  2. PII mask query before logging
  [Retrieve]  3. Embed query → hybrid retrieval (vector + lexical + RRF)
  [Rerank]    4. Cohere rerank top-N → top-k
  [Generate]  5. Claude synthesis with grounded citations
  [Guard-out] 6. Output groundedness check + PII re-mask on answer
"""

from __future__ import annotations

import anthropic
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_embedder, get_reranker, get_synthesizer
from app.api.schemas import QueryRequest, QueryResponse, SourceSchema
from app.config import Settings, get_settings
from app.generation.synthesizer import Synthesizer
from app.guardrails.output_guard import validate_output
from app.guardrails.pii import mask_pii
from app.guardrails.prompt_injection import check_injection
from app.indexing.embedder import Embedder
from app.logging import get_logger
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import Reranker

router = APIRouter(prefix="/v1", tags=["query"])
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse, summary="RAG query")
async def query(
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker),
    synthesizer: Synthesizer = Depends(get_synthesizer),
) -> QueryResponse:
    """Execute the full RAG pipeline and return a grounded, sanitized answer."""

    # ── Guard-in: prompt injection ────────────────────────────────────────────
    injection = check_injection(body.query)
    if injection.risk_level == "high":
        logger.warning("query_blocked_injection", risk=injection.risk_level)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query blocked: potential prompt injection detected.",
        )

    # ── Guard-in: PII mask the query before it appears in any log ────────────
    pii_scan = mask_pii(body.query)
    safe_query_for_log = pii_scan.masked_text
    logger.info("query_received", query=safe_query_for_log[:120])

    try:
        # ── Stage 1 + 2: hybrid retrieval (vector + lexical + RRF) ───────────
        fused = await hybrid_retrieve(body.query, embedder, top_n=body.top_n)

        # ── Stage 3: rerank ───────────────────────────────────────────────────
        ranked = await reranker.rerank(body.query, fused, top_k=body.top_k)

        # ── Stage 4: synthesize ───────────────────────────────────────────────
        result = await synthesizer.synthesize(
            query=body.query,
            ranked_results=ranked,
            max_tokens=body.max_tokens,
        )

    except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
        logger.error("generation_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation service temporarily unavailable. Please retry.",
        ) from exc
    except anthropic.AuthenticationError as exc:
        logger.error("anthropic_auth_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error.",
        ) from exc
    except Exception as exc:
        logger.error("query_pipeline_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        ) from exc

    # ── Guard-out: groundedness check + output PII re-mask ───────────────────
    guard = validate_output(result.answer, ranked)
    if guard.warnings:
        logger.warning("output_guard_warnings", warnings=guard.warnings)

    return QueryResponse(
        answer=guard.answer,
        sources=[
            SourceSchema(
                citation_number=s.citation_number,
                source_key=s.source_key,
                chunk_index=s.chunk_index,
                content_preview=s.content_preview,
            )
            for s in result.sources
        ],
        model=result.model,
    )
