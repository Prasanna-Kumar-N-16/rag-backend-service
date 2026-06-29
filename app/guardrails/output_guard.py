"""Output validation guardrail.

Two responsibilities:
1. **Groundedness check** — heuristically verify that the generated answer
   is anchored in the retrieved context (not hallucinated). Uses n-gram keyword
   overlap: if the answer shares too few content words with the context, it is
   flagged as potentially ungrounded.
2. **Output PII re-mask** — run the same PII masker over the final answer
   before it leaves the service, catching any PII that leaked from the context
   chunks into the generated text.

Both checks are designed to be fast and dependency-free; they trade recall for
latency so they can run inline on every request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.guardrails.pii import PIIResult, mask_pii
from app.logging import get_logger
from app.retrieval.reranker import RankedResult

logger = get_logger(__name__)

# Minimum fraction of non-trivial answer words that must appear in the context.
# Set conservatively (0.15) to flag clear hallucinations without false-positives
# on paraphrasing.
_MIN_GROUNDEDNESS_RATIO = 0.15

# Short answers (≤ this many tokens) skip the groundedness check.
_MIN_TOKENS_FOR_GROUNDEDNESS = 15

# Common words to exclude from keyword overlap calculation.
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "could should may might shall can cannot i you he she it we they them this "
    "that these those and or but not of in on at to for with by from as into "
    "through about against between into during before after above below up down "
    "out off over under again further then once here there when where why how "
    "all both each few more most other some such no only same so than too very".split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, excluding stopwords and short tokens."""
    return {
        w for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
        if w not in _STOPWORDS
    }


@dataclass
class OutputGuardResult:
    """Outcome of the output validation pass."""

    answer: str                          # final answer (PII-masked)
    is_grounded: bool
    groundedness_score: float            # 0.0 – 1.0
    pii_result: PIIResult
    warnings: list[str] = field(default_factory=list)


def validate_output(
    answer: str,
    context_chunks: list[RankedResult],
) -> OutputGuardResult:
    """Validate and sanitize a generated answer.

    Steps:
    1. PII-mask the answer.
    2. Compute keyword overlap between answer and retrieved context.
    3. Flag as ungrounded if overlap falls below threshold.

    Args:
        answer: Raw text from the generation step.
        context_chunks: The reranked chunks that were passed to generation.

    Returns:
        :class:`OutputGuardResult` with the sanitized answer and validation flags.
    """
    # Step 1: mask PII in the answer
    pii_result = mask_pii(answer)
    clean_answer = pii_result.masked_text

    warnings: list[str] = []
    if pii_result.has_pii:
        warnings.append(f"pii_redacted_from_output: {pii_result.found_types}")
        logger.warning("output_pii_redacted", types=pii_result.found_types)

    # Step 2: groundedness check
    answer_words = _tokenize(clean_answer)
    is_grounded = True
    groundedness_score = 1.0

    if len(answer_words) >= _MIN_TOKENS_FOR_GROUNDEDNESS and context_chunks:
        context_text = " ".join(c.content for c in context_chunks)
        context_words = _tokenize(context_text)

        if context_words:
            overlap = len(answer_words & context_words)
            groundedness_score = overlap / len(answer_words)
            is_grounded = groundedness_score >= _MIN_GROUNDEDNESS_RATIO

        if not is_grounded:
            warnings.append(
                f"low_groundedness_score: {groundedness_score:.2f} "
                f"(threshold {_MIN_GROUNDEDNESS_RATIO})"
            )
            logger.warning(
                "output_groundedness_low",
                score=groundedness_score,
                threshold=_MIN_GROUNDEDNESS_RATIO,
            )

    return OutputGuardResult(
        answer=clean_answer,
        is_grounded=is_grounded,
        groundedness_score=groundedness_score,
        pii_result=pii_result,
        warnings=warnings,
    )
