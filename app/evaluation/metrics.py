"""Pure, dependency-free evaluation metrics.

Every function here is deterministic and side-effect free so it can be unit
tested offline. The token-overlap approach mirrors
:mod:`app.guardrails.output_guard` so the offline faithfulness signal is
consistent with the inline guardrail that runs in production.

Retrieval metrics
-----------------
* ``recall_at_k``     — fraction of ground-truth sources that were retrieved.
* ``reciprocal_rank`` — 1 / rank of the first relevant source (0 if none).

Generation metrics
------------------
* ``answer_similarity`` — Jaccard token overlap vs. a reference answer.
* ``faithfulness``      — fraction of answer tokens grounded in the context.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "could should may might shall can cannot i you he she it we they them this "
    "that these those and or but not of in on at to for with by from as into "
    "through about against between during before after above below up down out "
    "off over under again further then once here there when where why how all "
    "both each few more most other some such no only same so than too very".split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, excluding stopwords and tokens under 3 chars."""
    return {w for w in re.findall(r"\b[a-z0-9]{3,}\b", text.lower()) if w not in _STOPWORDS}


def recall_at_k(
    retrieved_source_keys: Sequence[str],
    relevant_source_keys: Sequence[str],
) -> float:
    """Fraction of *relevant_source_keys* present in *retrieved_source_keys*.

    Returns ``0.0`` when there are no relevant keys (undefined recall); callers
    that want to exclude such examples from an average should check the ground
    truth before calling.
    """
    relevant = set(relevant_source_keys)
    if not relevant:
        return 0.0
    retrieved = set(retrieved_source_keys)
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_source_keys: Sequence[str],
    relevant_source_keys: Sequence[str],
) -> float:
    """Reciprocal rank of the first relevant source (0.0 if none retrieved)."""
    relevant = set(relevant_source_keys)
    for rank, key in enumerate(retrieved_source_keys, start=1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


def answer_similarity(generated: str, reference: str) -> float:
    """Jaccard token overlap between a generated and a reference answer (0–1)."""
    gen = _tokenize(generated)
    ref = _tokenize(reference)
    if not gen and not ref:
        return 1.0
    union = gen | ref
    if not union:
        return 0.0
    return len(gen & ref) / len(union)


def faithfulness(answer: str, context: str) -> float:
    """Fraction of the answer's content tokens that appear in the context (0–1).

    This is the offline analogue of the inline groundedness check in
    :func:`app.guardrails.output_guard.validate_output`. An answer with no
    content tokens is treated as fully grounded (1.0) — there is nothing
    unsupported to penalise.
    """
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = _tokenize(context)
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)
