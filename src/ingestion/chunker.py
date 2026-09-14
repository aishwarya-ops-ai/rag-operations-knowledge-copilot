from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

from src.models import Chunk, Document


def _split_oversized_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """Split a long paragraph at sentence boundaries, then hard-wrap if needed."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[start : start + chunk_size]
                for start in range(0, len(sentence), chunk_size)
            )
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)
    return pieces


def _paragraphs(text: str, chunk_size: int) -> Iterable[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            yield paragraph
        else:
            yield from _split_oversized_paragraph(paragraph, chunk_size)


def _overlap_tail(text: str, overlap: int) -> str:
    """Take an overlap tail, preferring to start at a word boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and " " in tail:
        tail = tail.split(" ", 1)[1]
    return tail.strip()


def chunk_document(
    document: Document,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    texts: List[str] = []
    current = ""

    for paragraph in _paragraphs(document.text, chunk_size):
        separator = "\n\n" if current else ""
        if len(current) + len(separator) + len(paragraph) <= chunk_size:
            current = f"{current}{separator}{paragraph}"
            continue

        if current:
            texts.append(current.strip())
            available_overlap = max(0, chunk_size - len(paragraph) - 2)
            tail = _overlap_tail(current, min(overlap, available_overlap))
            candidate = f"{tail}\n\n{paragraph}" if tail else paragraph
            current = candidate
        else:
            current = paragraph

    if current:
        texts.append(current.strip())

    chunks: List[Chunk] = []
    for index, text in enumerate(texts):
        digest = hashlib.sha256(
            f"{document.filename}:{index}:{text}".encode("utf-8")
        ).hexdigest()[:20]
        chunks.append(
            Chunk(
                id=digest,
                text=text,
                source=document.filename,
                chunk_index=index,
                character_count=len(text),
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
