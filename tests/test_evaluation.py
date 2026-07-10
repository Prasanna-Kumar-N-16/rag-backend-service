"""Tests for the golden-dataset evaluation module (offline, no live pipeline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.dataset import GoldenExample, load_golden_dataset
from app.evaluation.evaluator import evaluate
from app.evaluation.metrics import (
    answer_similarity,
    faithfulness,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.reranker import RankedResult

# ── Dataset loading ───────────────────────────────────────────────────────────

class TestLoadGoldenDataset:
    def test_loads_and_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "# a comment\n"
            "\n"
            '{"query": "q1", "reference_answer": "a1", "relevant_source_keys": ["k.txt"]}\n'
            '{"query": "q2"}\n',
            encoding="utf-8",
        )
        examples = load_golden_dataset(path)
        assert examples == [
            GoldenExample(query="q1", reference_answer="a1", relevant_source_keys=("k.txt",)),
            GoldenExample(query="q2", reference_answer=None, relevant_source_keys=()),
        ]

    def test_missing_query_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"reference_answer": "a"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing or empty 'query'"):
            load_golden_dataset(path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON on line 1"):
            load_golden_dataset(path)

    def test_shipped_example_dataset_is_valid(self) -> None:
        examples = load_golden_dataset("data/golden_dataset.example.jsonl")
        assert len(examples) == 3
        assert all(e.query for e in examples)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_recall_at_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "c"]) == 1.0
        assert recall_at_k(["a", "x"], ["a", "c"]) == 0.5
        assert recall_at_k(["x"], ["a"]) == 0.0
        assert recall_at_k(["a"], []) == 0.0  # no ground truth → 0

    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
        assert reciprocal_rank(["x", "b"], ["b"]) == 0.5
        assert reciprocal_rank(["x", "y"], ["z"]) == 0.0

    def test_answer_similarity(self) -> None:
        assert answer_similarity("Paris capital France", "Paris capital France") == 1.0
        assert answer_similarity("completely different words", "paris capital france") == 0.0
        assert 0.0 < answer_similarity("paris is the capital", "paris capital city") < 1.0

    def test_faithfulness(self) -> None:
        # Every content word of the answer appears in the context.
        assert faithfulness("Paris capital", "Paris is the capital of France") == 1.0
        # Half the answer's content words are ungrounded.
        assert faithfulness("Paris Tokyo", "Paris is in France") == 0.5
        # Empty/stopword-only answer is vacuously grounded.
        assert faithfulness("the is a", "anything") == 1.0


# ── Evaluator ─────────────────────────────────────────────────────────────────

def _chunk(source_key: str, content: str) -> RankedResult:
    return RankedResult(
        id=source_key,
        source_key=source_key,
        chunk_index=0,
        content=content,
        relevance_score=0.9,
    )


class TestEvaluate:
    async def test_end_to_end_scoring_with_fakes(self) -> None:
        dataset = [
            GoldenExample(
                query="What is the capital of France?",
                reference_answer="Paris is the capital of France.",
                relevant_source_keys=("france.txt",),
            ),
            GoldenExample(query="Unlabelled query"),  # no ground truth
        ]

        async def retrieve(query: str) -> list[RankedResult]:
            if "capital" in query:
                return [_chunk("france.txt", "Paris is the capital of France.")]
            return [_chunk("other.txt", "Some unrelated context about widgets.")]

        async def synthesize(query: str, chunks: list[RankedResult]) -> str:
            return chunks[0].content if chunks else ""

        report = await evaluate(dataset, retrieve, synthesize)

        assert report.count == 2
        first = report.results[0]
        assert first.recall_at_k == 1.0
        assert first.reciprocal_rank == 1.0
        assert first.faithfulness == 1.0
        assert first.answer_similarity is not None and first.answer_similarity > 0.5

        # The unlabelled example has no recall/similarity but still scores faithfulness.
        second = report.results[1]
        assert second.recall_at_k is None
        assert second.answer_similarity is None
        assert second.faithfulness == 1.0

        # Aggregates only average over examples that carry the relevant label.
        assert report.mean_recall_at_k == 1.0
        assert report.mean_faithfulness == 1.0
