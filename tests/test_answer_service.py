from src.config import INSUFFICIENT_ANSWER
from src.generation.answer_service import AnswerService
from src.generation.provider import GenerationUnavailableError
from src.models import RetrievalResult


def evidence(similarity=0.8, source="policy.txt", chunk_index=2):
    return RetrievalResult(
        chunk_id="chunk-id",
        text="Team Leads may approve refunds up to $750.",
        source=source,
        chunk_index=chunk_index,
        distance=1.0 - similarity,
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, question, top_k):
        assert question
        return self.results[:top_k]


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, system_instruction, user_prompt):
        self.calls.append((system_instruction, user_prompt))
        return self.response


class SequenceProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, system_instruction, user_prompt):
        self.calls.append((system_instruction, user_prompt))
        return next(self.responses)


class UnavailableProvider:
    def generate(self, system_instruction, user_prompt):
        raise GenerationUnavailableError("not running")


def test_supported_answer_keeps_valid_citation_and_source():
    provider = FakeProvider("A Team Lead may approve up to $750. [E1]")
    service = AnswerService(FakeRetriever([evidence()]), provider)

    result = service.answer("What is the approval limit?", top_k=4)

    assert result.status == "grounded"
    assert result.confidence == "High"
    assert "0.800 met the minimum threshold 0.50" in result.decision_reason
    assert len(result.sources) == 1
    assert result.sources[0].evidence_id == "E1"
    assert result.sources[0].filename == "policy.txt"
    assert result.sources[0].chunk_index == 2
    assert result.cited_evidence_ids == ("E1",)
    assert result.evidence[0].text.startswith("Team Leads")
    assert "Answer only from the evidence" in provider.calls[0][0]
    assert "must contain that exact value" in provider.calls[0][0]
    assert "Source: policy.txt" in provider.calls[0][1]


def test_low_similarity_returns_insufficient_without_calling_model():
    provider = FakeProvider("This must not be used. [E1]")
    service = AnswerService(FakeRetriever([evidence(similarity=0.3)]), provider)

    result = service.answer("What is the holiday policy?", top_k=4)

    assert result.status == "insufficient"
    assert result.confidence == "Insufficient evidence"
    assert result.answer == INSUFFICIENT_ANSWER
    assert "0.300 is below the minimum threshold 0.50" in result.decision_reason
    assert result.sources == ()
    assert result.evidence
    assert provider.calls == []


def test_medium_confidence_is_visible_between_thresholds():
    provider = FakeProvider("A Team Lead may approve up to $750. [E1]")
    service = AnswerService(FakeRetriever([evidence(similarity=0.6)]), provider)

    result = service.answer("What is the approval limit?", top_k=4)

    assert result.status == "grounded"
    assert result.confidence == "Medium"


def test_no_retrieved_evidence_returns_insufficient():
    provider = FakeProvider("This must not be used. [E1]")
    service = AnswerService(FakeRetriever([]), provider)

    result = service.answer("Unknown question", top_k=4)

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.confidence == "Insufficient evidence"
    assert result.decision_reason == "No evidence chunks were retrieved."
    assert provider.calls == []


def test_model_refusal_is_normalized_to_required_message():
    provider = FakeProvider(f"  {INSUFFICIENT_ANSWER}  ")
    service = AnswerService(FakeRetriever([evidence()]), provider)

    result = service.answer("Unsupported question", top_k=4)

    assert result.status == "insufficient"
    assert result.confidence == "Insufficient evidence"
    assert result.answer == INSUFFICIENT_ANSWER
    assert "local model reported" in result.decision_reason


def test_uncited_model_answer_is_rejected():
    provider = FakeProvider("A Team Lead may approve up to $750.")
    service = AnswerService(FakeRetriever([evidence()]), provider)

    result = service.answer("What is the approval limit?", top_k=4)

    assert result.status == "insufficient"
    assert result.answer == INSUFFICIENT_ANSWER
    assert "no evidence citations" in result.decision_reason


def test_unknown_evidence_citation_is_rejected():
    provider = FakeProvider("A Team Lead may approve up to $750. [E7]")
    service = AnswerService(FakeRetriever([evidence()]), provider)

    result = service.answer("What is the approval limit?", top_k=4)

    assert result.status == "insufficient"
    assert result.sources == ()
    assert "was not retrieved" in result.decision_reason


def test_numeric_claim_with_wrong_citation_gets_one_corrective_retry():
    results = [
        evidence(source="refunds.txt", chunk_index=2),
        RetrievalResult(
            chunk_id="other",
            text="Requests submitted after 60 days require manager approval.",
            source="refunds.txt",
            chunk_index=1,
            distance=0.3,
        ),
    ]
    provider = SequenceProvider(
        [
            "A Team Lead may approve up to $750 [E2].",
            "A Team Lead may approve up to $750 [E1].",
        ]
    )
    service = AnswerService(FakeRetriever(results), provider)

    result = service.answer("What is the approval limit?", top_k=4)

    assert result.status == "grounded"
    assert result.cited_evidence_ids == ("E1",)
    assert len(provider.calls) == 2
    assert "exact numeric claims: $750" in provider.calls[1][1]
    assert "[E1]" in provider.calls[1][1]


def test_unsupported_numeric_citation_is_rejected_after_retry():
    results = [
        evidence(source="refunds.txt", chunk_index=2),
        RetrievalResult(
            chunk_id="other",
            text="Requests submitted after 60 days require manager approval.",
            source="refunds.txt",
            chunk_index=1,
            distance=0.3,
        ),
    ]
    provider = SequenceProvider(
        [
            "A Team Lead may approve up to $750 [E2].",
            "A Team Lead may approve up to $750 [E2].",
        ]
    )
    service = AnswerService(FakeRetriever(results), provider)

    result = service.answer("What is the approval limit?", top_k=4)

    assert result.status == "insufficient"
    assert result.answer == INSUFFICIENT_ANSWER
    assert "exact numeric claims" in result.decision_reason


def test_sources_are_deduplicated_in_citation_order():
    results = [
        evidence(source="refunds.txt", chunk_index=1),
        evidence(source="refunds.txt", chunk_index=2),
        evidence(source="escalations.txt", chunk_index=1),
    ]
    provider = FakeProvider("Refund rule [E2]. Escalation rule [E3] and [E2].")
    service = AnswerService(FakeRetriever(results), provider)

    result = service.answer("Explain both rules", top_k=4)

    assert result.status == "grounded"
    assert result.cited_evidence_ids == ("E2", "E3")
    assert [source.evidence_id for source in result.sources] == ["E2", "E3"]
    assert [source.filename for source in result.sources] == [
        "refunds.txt",
        "escalations.txt",
    ]
    assert [source.chunk_index for source in result.sources] == [2, 1]


def test_unavailable_local_model_keeps_retrieved_evidence_visible():
    service = AnswerService(FakeRetriever([evidence()]), UnavailableProvider())

    result = service.answer("What is the approval limit?", top_k=4)

    assert result.status == "generation_unavailable"
    assert result.confidence == "Insufficient evidence"
    assert result.answer == INSUFFICIENT_ANSWER
    assert result.evidence[0].source == "policy.txt"
    assert result.sources == ()
