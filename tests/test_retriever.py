import pytest

from src.models import RetrievalResult
from src.retrieval.retriever import Retriever


class FakeEmbedder:
    def encode(self, texts):
        assert list(texts) == ["refund timing"]
        return [[1.0, 0.0]]


class FakeStore:
    def query(self, query_embedding, top_k):
        assert query_embedding == [1.0, 0.0]
        assert top_k == 2
        return [
            RetrievalResult(
                chunk_id="one",
                text="Refund timing details",
                source="refunds.txt",
                chunk_index=3,
                distance=0.2,
            )
        ]


def test_retriever_embeds_query_and_returns_results():
    retriever = Retriever(FakeEmbedder(), FakeStore())
    result = retriever.search(" refund timing ", top_k=2)[0]

    assert result.source == "refunds.txt"
    assert result.similarity == pytest.approx(0.8)


def test_empty_query_is_rejected():
    retriever = Retriever(FakeEmbedder(), FakeStore())
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.search("   ", top_k=2)

