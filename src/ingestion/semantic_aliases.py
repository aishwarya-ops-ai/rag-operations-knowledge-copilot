"""Conservative, search-only aliases for concepts present in source chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable, List, Pattern, Tuple

from src.models import Chunk


@dataclass(frozen=True)
class AliasRule:
    pattern: Pattern[str]
    canonical_query: str
    aliases: Tuple[str, ...]


ALIAS_RULES = (
    AliasRule(
        re.compile(r"\bcheck[\s-]?in\b", re.I),
        "employee shift check-in attendance procedure",
        ("clock in", "punch in"),
    ),
    AliasRule(
        re.compile(r"\b(?:forgot to check[\s-]?in|retroactive check[\s-]?in)\b", re.I),
        "employee forgot to check in retroactive correction",
        ("forgot to clock in", "missed check-in", "failed to check in"),
    ),
    AliasRule(
        re.compile(r"\bno-call no-show\b", re.I),
        "employee no-call no-show attendance procedure",
        ("missed shift without notice", "did not show up for shift"),
    ),
    AliasRule(
        re.compile(r"\bcustomer\b[^.]{0,180}\b(?:supervisor|escalat\w*)\b", re.I),
        "customer requests supervisor escalation policy",
        ("speak to a supervisor", "request a manager", "customer escalation"),
    ),
    AliasRule(
        re.compile(r"\brefund(?:s|ed|ing)?\b", re.I),
        "customer refund policy",
        ("money back", "return customer payment", "customer refund request"),
    ),
    AliasRule(
        re.compile(r"\bservice credit(?:s)?\b", re.I),
        "customer service credit policy",
        ("account credit", "future service credit"),
    ),
    AliasRule(
        re.compile(r"\b(?:quality review|quality score|quality appeal)\b", re.I),
        "employee quality review guidelines",
        ("QA review", "quality assurance review", "QA score"),
    ),
    AliasRule(
        re.compile(r"\b(?:new employee training|operations onboarding|trainee)\b", re.I),
        "new employee training procedure",
        ("new hire onboarding", "employee onboarding", "new starter training"),
    ),
)


def aliases_for_text(text: str) -> Tuple[str, ...]:
    """Return aliases only for concepts explicitly present in the chunk."""
    aliases: List[str] = []
    seen = set()
    for rule in ALIAS_RULES:
        if not rule.pattern.search(text):
            continue
        for alias in rule.aliases:
            key = alias.casefold()
            if key not in seen:
                aliases.append(alias)
                seen.add(key)
    return tuple(aliases)


def add_retrieval_aliases(chunks: Iterable[Chunk]) -> List[Chunk]:
    """Attach search metadata while retaining each chunk's original text."""
    return [
        replace(chunk, retrieval_aliases=aliases_for_text(chunk.text))
        for chunk in chunks
    ]


def _normalized_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def alias_query_variants(query: str) -> List[str]:
    """Map exact known aliases to canonical operational retrieval queries."""
    normalized_query = f" {_normalized_phrase(query)} "
    variants: List[str] = []
    seen = set()
    for rule in ALIAS_RULES:
        if not any(
            f" {_normalized_phrase(alias)} " in normalized_query
            for alias in rule.aliases
        ):
            continue
        key = rule.canonical_query.casefold()
        if key not in seen:
            variants.append(rule.canonical_query)
            seen.add(key)
    return variants
