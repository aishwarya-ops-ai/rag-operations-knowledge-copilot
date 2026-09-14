import pytest

from src.models import RetrievalResult
from src.retrieval.query_normalizer import normalize_query, semantic_query_variants
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


@pytest.mark.parametrize(
    "query, expected_variant",
    [
        (
            "what if I forgot to check in",
            "employee forgot to check in retroactive correction",
        ),
        (
            "I missed check-in, what should I do?",
            "employee missed start-of-shift check-in attendance procedure",
        ),
        (
            "what happens if someone doesn’t check in?",
            "employee missed start-of-shift check-in attendance procedure",
        ),
        (
            "forgot to clock in",
            "employee forgot to check in retroactive correction",
        ),
        (
            "missed the start-of-shift check-in",
            "employee missed start-of-shift check-in attendance procedure",
        ),
    ],
)
def test_missed_check_in_wording_adds_narrow_semantic_variant(
    query, expected_variant
):
    assert semantic_query_variants(query) == [query, expected_variant]


@pytest.mark.parametrize(
    "query",
    [
        "can I get my money back?",
        "Could I receive a refund?",
        "I want a refund",
    ],
)
def test_casual_refund_wording_adds_meaning_preserving_variant(query):
    normalization = normalize_query(query)

    assert normalization is not None
    assert normalization.rule == "casual_refund_request"
    assert semantic_query_variants(query) == [query, "customer requests a refund"]


@pytest.mark.parametrize(
    "query",
    [
        "what is the maternity leave policy?",
        "who is the CEO?",
        "what is the travel allowance?",
        "How should an employee check in normally?",
        "I want to speak to someone",
        "How much can a Team Lead approve for a refund?",
    ],
)
def test_unrelated_or_non_missed_queries_are_not_expanded(query):
    assert semantic_query_variants(query) == [query]


class VariantEmbedder:
    def encode(self, texts):
        self.queries = list(texts)
        return [[0.0, 1.0], [1.0, 0.0]]


class VariantStore:
    def query(self, query_embedding, top_k):
        if query_embedding == [0.0, 1.0]:
            return [
                RetrievalResult(
                    chunk_id="general",
                    text="General attendance procedure",
                    source="attendance.txt",
                    chunk_index=1,
                    distance=0.62,
                )
            ]
        return [
            RetrievalResult(
                chunk_id="forgotten-check-in",
                text="Retroactive check-in procedure",
                source="attendance.txt",
                chunk_index=6,
                distance=0.44,
            )
        ]


def test_semantic_variant_can_promote_the_supported_chunk():
    embedder = VariantEmbedder()
    retriever = Retriever(embedder, VariantStore())

    results = retriever.search("what if I forgot to check in", top_k=4)

    assert embedder.queries == [
        "what if I forgot to check in",
        "employee forgot to check in retroactive correction",
    ]
    assert results[0].chunk_id == "forgotten-check-in"
    assert results[0].similarity == pytest.approx(0.56)


class DisagreeingVariantStore(VariantStore):
    def query(self, query_embedding, top_k):
        results = super().query(query_embedding, top_k)
        if query_embedding == [1.0, 0.0]:
            result = results[0]
            return [
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    source="different-policy.txt",
                    chunk_index=result.chunk_index,
                    distance=result.distance,
                )
            ]
        return results


def test_semantic_variant_is_rejected_when_top_sources_disagree():
    retriever = Retriever(VariantEmbedder(), DisagreeingVariantStore())

    results = retriever.search("what if I forgot to check in", top_k=4)

    assert [result.chunk_id for result in results] == ["general"]
    assert results[0].similarity == pytest.approx(0.38)
