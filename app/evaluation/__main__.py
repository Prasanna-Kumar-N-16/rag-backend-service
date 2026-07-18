"""CLI: evaluate the live RAG pipeline against a golden dataset.

Usage
-----
    python -m app.evaluation --dataset data/golden_dataset.example.jsonl

Requires the same configuration as the service (database URL + provider keys),
since it drives the real retrieval and generation stack. Prints a JSON summary
to stdout and exits non-zero if mean faithfulness falls below ``--min-faithfulness``
so it can double as a regression gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import get_settings
from app.db import close_db, init_db
from app.evaluation.dataset import load_golden_dataset
from app.evaluation.evaluator import EvaluationReport, evaluate
from app.generation.claude_client import ClaudeClient
from app.generation.synthesizer import Synthesizer
from app.indexing.embedder import Embedder
from app.logging import configure_logging
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import RankedResult, Reranker


async def _run(dataset_path: str) -> EvaluationReport:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    dataset = load_golden_dataset(dataset_path)
    embedder = Embedder(settings)
    reranker = Reranker(settings)
    synthesizer = Synthesizer(
        client=ClaudeClient(settings), model=settings.generation_model
    )

    async def retrieve(query: str) -> list[RankedResult]:
        fused = await hybrid_retrieve(query, embedder, top_n=settings.retrieval_top_n)
        return await reranker.rerank(query, fused, top_k=settings.rerank_top_k)

    async def synthesize(query: str, chunks: list[RankedResult]) -> str:
        result = await synthesizer.synthesize(query=query, ranked_results=chunks)
        return result.answer

    await init_db(settings)
    try:
        return await evaluate(dataset, retrieve, synthesize)
    finally:
        await close_db()


def _summary(report: EvaluationReport) -> dict[str, object]:
    return {
        "examples": report.count,
        "mean_faithfulness": report.mean_faithfulness,
        "mean_recall_at_k": report.mean_recall_at_k,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "mean_answer_similarity": report.mean_answer_similarity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL golden dataset.")
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.0,
        help="Exit non-zero if mean faithfulness is below this threshold.",
    )
    args = parser.parse_args()

    report = asyncio.run(_run(args.dataset))
    summary = _summary(report)
    print(json.dumps(summary, indent=2))

    mean = report.mean_faithfulness
    if mean is not None and mean < args.min_faithfulness:
        print(
            f"FAIL: mean faithfulness {mean:.3f} < threshold {args.min_faithfulness}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
