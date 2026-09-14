from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Document:
    path: Path
    filename: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    character_count: int
    retrieval_aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    distance: float

    @property
    def similarity(self) -> float:
        """Convert Chroma cosine distance to cosine similarity."""
        return 1.0 - self.distance


@dataclass(frozen=True)
class SourceReference:
    evidence_id: str
    filename: str
    chunk_index: int


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    confidence: str
    decision_reason: str
    sources: Tuple[SourceReference, ...]
    evidence: Tuple[RetrievalResult, ...]
    cited_evidence_ids: Tuple[str, ...]
    status: str
