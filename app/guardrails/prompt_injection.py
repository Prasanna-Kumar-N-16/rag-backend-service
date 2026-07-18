"""Prompt injection defense.

Detects common prompt injection and jailbreak patterns in user queries before
they reach the RAG pipeline. Combines two layers:

1. **Denylist patterns** — compiled regexes for well-known injection phrases
   ("ignore previous instructions", role-override attempts, "DAN" jailbreaks).
2. **Heuristic signals** — structural indicators that suggest meta-instruction
   rather than a genuine information-seeking query (e.g. excessive ALL-CAPS
   commands, repeated override keywords).

The detector is intentionally conservative: it returns a detection result with
matched evidence so callers can decide whether to block, log, or allow with
reduced trust. It does *not* make that policy decision itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.logging import get_logger
from app.retrieval.reranker import RankedResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Denylist — compiled patterns for known injection phrases
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,30}\b(?:previous|above|prior|all)?\b.{0,20}\b(?:instructions?|rules?|constraints?|guidelines?|system|prompt)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    ("role_override", re.compile(
        r"\b(?:you are now|act as|pretend (?:you are|to be)|behave as"
        r"|roleplay as|simulate)\b.{0,40}"
        r"\b(?:an? |the )?(?:evil|unfiltered|uncensored|unrestricted|jailbroken|DAN)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    ("system_override", re.compile(
        r"\[?\b(?:SYSTEM|ASSISTANT|HUMAN|USER)\b\]?\s*:?\s*(?:ignore|you must|always|never)\b",
        re.IGNORECASE,
    )),
    ("jailbreak_dan", re.compile(
        r"\bDAN\b|\bdo anything now\b|\bjailbreak\b|\bunfiltered mode\b|\bdev(?:eloper)? mode\b",
        re.IGNORECASE,
    )),
    ("leaked_instructions", re.compile(
        r"\b(?:repeat|print|reveal|show|output|display|return)\b.{0,30}"
        r"\b(?:system prompt|instructions?|context|above text|exact words)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    ("exfiltration_attempt", re.compile(
        r"\b(?:translate|summarize|encode|base64|hex)\b.{0,20}\b(?:system|prompt|instructions?|context|rules?)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    ("injection_delimiters", re.compile(
        r"(?:```|\[\[|\]\]|<\||\|>)\s*(?:system|instruction|prompt|override)\b",
        re.IGNORECASE,
    )),
]

# Heuristic thresholds
_MAX_CAPS_RATIO = 0.40       # more than 40 % uppercase letters → suspicious
_MAX_REPEAT_KEYWORDS = 3     # same injection keyword appearing > 3 times


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    """Outcome of a prompt-injection scan."""

    is_injection: bool
    matched_patterns: list[str] = field(default_factory=list)
    heuristic_flags: list[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        total = len(self.matched_patterns) + len(self.heuristic_flags)
        if total == 0:
            return "none"
        if total == 1:
            return "low"
        if total <= 3:
            return "medium"
        return "high"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_injection(text: str) -> InjectionResult:
    """Scan *text* for prompt injection indicators.

    Args:
        text: The user query to evaluate.

    Returns:
        :class:`InjectionResult` with detection flag, matched patterns, and
        heuristic flags. Does not raise.
    """
    matched: list[str] = []
    heuristics: list[str] = []

    # Pattern matching
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(name)

    # Heuristic: excessive ALL-CAPS (ignoring short texts)
    if len(text) > 20:
        letters = [c for c in text if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > _MAX_CAPS_RATIO:
            heuristics.append("excessive_caps")

    # Heuristic: repeated override keywords
    override_words = re.findall(
        r"\b(?:ignore|override|bypass|disregard|forget)\b", text, re.IGNORECASE
    )
    if len(override_words) > _MAX_REPEAT_KEYWORDS:
        heuristics.append("repeated_override_keywords")

    is_injection = bool(matched or heuristics)

    if is_injection:
        logger.warning(
            "prompt_injection_detected",
            patterns=matched,
            heuristics=heuristics,
            risk=InjectionResult(is_injection=True, matched_patterns=matched,
                                 heuristic_flags=heuristics).risk_level,
        )

    return InjectionResult(
        is_injection=is_injection,
        matched_patterns=matched,
        heuristic_flags=heuristics,
    )


def check_context_injection(chunks: list[RankedResult]) -> InjectionResult:
    """Scan retrieved context chunks for prompt injection indicators.

    Defends against indirect (second-order) injection: a poisoned source
    document whose content carries meta-instructions that could hijack the
    generation step once it reaches the Claude prompt. This is a distinct
    attack surface from :func:`check_injection`, which only sees the user's
    own query text.

    Args:
        chunks: Reranked context chunks about to be sent to generation.

    Returns:
        :class:`InjectionResult` for the concatenated chunk text. An empty
        chunk list returns a clean (non-injection) result.
    """
    if not chunks:
        return InjectionResult(is_injection=False)
    combined = "\n\n".join(chunk.content for chunk in chunks)
    return check_injection(combined)
