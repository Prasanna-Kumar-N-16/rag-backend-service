"""Golden-dataset loading.

A golden dataset is a JSONL file where each line is one evaluation example:

    {"query": "...", "reference_answer": "...", "relevant_source_keys": ["a.txt"]}

``reference_answer`` and ``relevant_source_keys`` are both optional — an example
with neither still exercises retrieval + generation and yields a faithfulness
score (which needs no ground truth). Blank lines and ``#`` comment lines are
ignored so datasets can be annotated inline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldenExample:
    """A single labelled evaluation example."""

    query: str
    reference_answer: str | None = None
    relevant_source_keys: tuple[str, ...] = ()


def load_golden_dataset(path: str | Path) -> list[GoldenExample]:
    """Load and validate a JSONL golden dataset from *path*.

    Args:
        path: Filesystem path to a ``.jsonl`` file.

    Returns:
        Ordered list of :class:`GoldenExample`.

    Raises:
        ValueError: On malformed JSON or a line missing the required ``query``.
    """
    examples: list[GoldenExample] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            query = obj.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Line {line_no}: missing or empty 'query'.")

            reference = obj.get("reference_answer")
            examples.append(
                GoldenExample(
                    query=query,
                    reference_answer=str(reference) if reference is not None else None,
                    relevant_source_keys=tuple(obj.get("relevant_source_keys", [])),
                )
            )
    return examples
