from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_document
from src.models import Document


def make_document(text: str) -> Document:
    return Document(path=Path("policy.txt"), filename="policy.txt", text=text)


def test_chunks_respect_size_and_keep_metadata():
    document = make_document("A" * 120 + "\n\n" + "B" * 120 + "\n\n" + "C" * 120)
    chunks = chunk_document(document, chunk_size=200, overlap=30)

    assert len(chunks) == 3
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.source == "policy.txt" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_overlap_is_carried_into_next_chunk():
    first = "alpha beta gamma delta epsilon zeta eta theta"
    second = "new paragraph that cannot fit into the first chunk"
    chunks = chunk_document(make_document(f"{first}\n\n{second}"), chunk_size=70, overlap=25)

    assert len(chunks) == 2
    assert "zeta eta theta" in chunks[1].text
    assert second in chunks[1].text


def test_invalid_chunk_configuration():
    with pytest.raises(ValueError):
        chunk_document(make_document("text"), chunk_size=100, overlap=100)

