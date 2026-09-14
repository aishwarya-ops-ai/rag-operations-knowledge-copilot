"""Deterministic, meaning-preserving query variants for local retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import List, Optional


LOGGER = logging.getLogger(__name__)

CHECK_IN_ACTION_PATTERN = re.compile(r"\b(?:check[\s-]?in|clock[\s-]?in)\b", re.I)
FORGOTTEN_ACTION_PATTERN = re.compile(
    r"\b(?:forget|forgets|forgot|forgotten)\b",
    re.I,
)
MISSED_ACTION_PATTERN = re.compile(
    r"\b(?:miss|missed|missing|doesn['’]?t|does not|didn['’]?t|did not|"
    r"fail|failed)\b",
    re.I,
)
CASUAL_REFUND_PATTERN = re.compile(
    r"\bmoney\s+back\b|"
    r"\b(?:can|could|may)\s+(?:i|we)\s+(?:get|receive)\s+(?:a\s+)?refund\b|"
    r"\b(?:i|we)\s+(?:want|need)\s+(?:a\s+)?refund\b",
    re.I,
)

FORGOTTEN_CHECK_IN_QUERY = "employee forgot to check in retroactive correction"
MISSED_CHECK_IN_QUERY = "employee missed start-of-shift check-in attendance procedure"
REFUND_REQUEST_QUERY = "customer requests a refund"


@dataclass(frozen=True)
class QueryNormalization:
    text: str
    rule: str


def normalize_query(query: str) -> Optional[QueryNormalization]:
    """Return one conservative operational variant, or None when intent is unclear."""
    if CHECK_IN_ACTION_PATTERN.search(query):
        if FORGOTTEN_ACTION_PATTERN.search(query):
            return QueryNormalization(
                text=FORGOTTEN_CHECK_IN_QUERY,
                rule="forgotten_check_in",
            )
        if MISSED_ACTION_PATTERN.search(query):
            return QueryNormalization(
                text=MISSED_CHECK_IN_QUERY,
                rule="missed_check_in",
            )

    if CASUAL_REFUND_PATTERN.search(query):
        return QueryNormalization(
            text=REFUND_REQUEST_QUERY,
            rule="casual_refund_request",
        )

    return None


def semantic_query_variants(query: str) -> List[str]:
    """Keep the original query and append at most one deterministic variant."""
    variants = [query]
    normalization = normalize_query(query)
    if normalization and query.casefold() != normalization.text.casefold():
        LOGGER.debug("Applied query normalization rule: %s", normalization.rule)
        variants.append(normalization.text)
    return variants
