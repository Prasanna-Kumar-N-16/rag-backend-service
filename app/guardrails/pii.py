"""PII detection and masking.

Uses a lightweight regex-based detector that covers the most common PII
categories without requiring any external ML model. If the optional
``presidio-analyzer`` + ``presidio-anonymizer`` packages are installed (via
``pip install ".[pii]"``) a richer detection pass is added on top.

Masked tokens use a stable placeholder format: ``[REDACTED:<TYPE>]`` so that
downstream consumers can distinguish different entity types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns (compiled once at module load)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Email
    ("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        re.IGNORECASE,
    )),
    # US phone (various formats)
    ("PHONE", re.compile(
        r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
    )),
    # US SSN
    ("SSN", re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    )),
    # Credit/debit card (Luhn-ignorant, 13-19 digits with optional spaces/dashes)
    ("CREDIT_CARD", re.compile(
        r"\b(?:\d[ \-]?){13,19}\b",
    )),
    # IPv4
    ("IP_ADDRESS", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    )),
    # US street address (heuristic)
    ("ADDRESS", re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]{3,40}(?:St|Ave|Blvd|Dr|Rd|Ln|Ct|Way|Pl)\b\.?",
        re.IGNORECASE,
    )),
    # API / secret keys (common patterns: long hex/base64 strings)
    ("API_KEY", re.compile(
        r"\b(?:sk|pk|api|token|key)[-_]?[A-Za-z0-9]{20,}\b",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PIIResult:
    """Outcome of a PII scan on a single text."""

    masked_text: str
    found_types: list[str] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.found_types)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mask_pii(text: str) -> PIIResult:
    """Detect and mask PII in *text*.

    Applies the built-in regex patterns. If ``presidio`` is available it runs
    a second pass to catch names, dates-of-birth, and other NER-based entities.

    Args:
        text: The input string to scan.

    Returns:
        :class:`PIIResult` with the masked text and the list of detected types.
    """
    masked = text
    found: list[str] = []

    for entity_type, pattern in _PATTERNS:
        new_text, count = pattern.subn(f"[REDACTED:{entity_type}]", masked)
        if count:
            masked = new_text
            found.append(entity_type)

    # Optional presidio pass (heavier, covers names, DOB, location, etc.)
    masked, presidio_types = _presidio_pass(masked)
    found.extend(presidio_types)

    if found:
        logger.debug("pii_detected", types=found)

    return PIIResult(masked_text=masked, found_types=list(dict.fromkeys(found)))


def _presidio_pass(text: str) -> tuple[str, list[str]]:
    """Run Presidio analyzer + anonymizer if the package is available."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return text, []

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text, []

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    entity_types = list({r.entity_type for r in results})
    return anonymized.text, entity_types
