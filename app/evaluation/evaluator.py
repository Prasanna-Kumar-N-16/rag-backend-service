"""Evaluation orchestration over a golden dataset.

The evaluator is decoupled from the live pipeline: it takes two async callables
(``retrieve`` and ``synthesize``) so it can run against the real
retrieval/generation stack in production *or* against fakes in tests. The CLI
in :mod:`app.evaluation.__main__` wires the real implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.evaluation.dataset import GoldenExample
from app.evaluation.metrics import (
    answer_similarity,
    faithfulness,
    recall_at_k,
    reciprocal_rank,
)
from app.logging import get_logger
from app.retrieval.reranker import RankedResult

logger = get_logger(__name__)


class RetrieveFn(Protocol):
    """Runs the retrieval + rerank stack for a query, returning ranked chunks."""

    async def __call__(self, query: str) -> list[RankedResult]: ...


class SynthesizeFn(Protocol):
    """Generates a grounded answer from a query and its ranked chunks."""

    async def __call__(self, query: str, chunks: list[RankedResult]) -> str: ...


@dataclass(frozen=True)
class ExampleResult:
    """Per-example evaluation outcome. Ground-truth metrics are ``None`` when
    the example carries no corresponding label."""

    query: str
    generated_answer: str
    retrieved_source_keys: list[str]
    faithfulness: float
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    answer_similarity: float | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate report over all evaluated examples."""

    results: list[ExampleResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.results)

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def mean_faithfulness(self) -> float | None:
        return self._mean([r.faithfulness for r in self.results])

    @property
    def mean_recall_at_k(self) -> float | None:
        return self._mean([r.recall_at_k for r in self.results if r.recall_at_k is not None])

    @property
    def mean_reciprocal_rank(self) -> float | None:
        return self._mean(
            [r.reciprocal_rank for r in self.results if r.reciprocal_rank is not None]
        )

    @property
    def mean_answer_similarity(self) -> float | None:
        return self._mean(
            [r.answer_similarity for r in self.results if r.answer_similarity is not None]
        )


async def evaluate(
    dataset: list[GoldenExample],
    retrieve: RetrieveFn,
    synthesize: SynthesizeFn,
) -> EvaluationReport:
    """Run *retrieve* → *synthesize* over *dataset* and score each example.

    Args:
        dataset: Golden examples to evaluate.
        retrieve: Async callable mapping a query to reranked chunks.
        synthesize: Async callable mapping (query, chunks) to an answer string.

    Returns:
        An :class:`EvaluationReport` with per-example results and aggregate means.
    """
    results: list[ExampleResult] = []

    for example in dataset:
        chunks = await retrieve(example.query)
        answer = await synthesize(example.query, chunks)

        retrieved_keys = [c.source_key for c in chunks]
        context = "\n\n".join(c.content for c in chunks)

        recall: float | None = None
        rr: float | None = None
        if example.relevant_source_keys:
            recall = recall_at_k(retrieved_keys, example.relevant_source_keys)
            rr = reciprocal_rank(retrieved_keys, example.relevant_source_keys)

        similarity: float | None = None
        if example.reference_answer is not None:
            similarity = answer_similarity(answer, example.reference_answer)

        results.append(
            ExampleResult(
                query=example.query,
                generated_answer=answer,
                retrieved_source_keys=retrieved_keys,
                faithfulness=faithfulness(answer, context),
                recall_at_k=recall,
                reciprocal_rank=rr,
                answer_similarity=similarity,
            )
        )

    report = EvaluationReport(results=results)
    logger.info(
        "evaluation_done",
        examples=report.count,
        mean_faithfulness=report.mean_faithfulness,
        mean_recall_at_k=report.mean_recall_at_k,
        mean_reciprocal_rank=report.mean_reciprocal_rank,
        mean_answer_similarity=report.mean_answer_similarity,
    )
    return report
