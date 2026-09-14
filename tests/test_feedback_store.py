import json

import pytest

from src.feedback.store import FeedbackStore
from src.models import GroundedAnswer, RetrievalResult, SourceReference


def grounded_answer():
    evidence = RetrievalResult(
        chunk_id="refund-chunk",
        text="Team Leads may approve refunds up to $750.",
        source="refunds.txt",
        chunk_index=2,
        distance=0.2,
    )
    return GroundedAnswer(
        answer="A Team Lead may approve up to $750. [E1]",
        confidence="High",
        decision_reason="Similarity and citations passed.",
        sources=(SourceReference("E1", "refunds.txt", 2),),
        evidence=(evidence,),
        cited_evidence_ids=("E1",),
        status="grounded",
    )


def test_records_feedback_as_append_only_jsonl(tmp_path):
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path)

    first = store.record_answer(
        "What is the refund limit?",
        grounded_answer(),
        top_k=4,
        rating="helpful",
        comment="Correct and clear",
    )
    store.record_answer(
        "What is the refund limit?",
        grounded_answer(),
        top_k=4,
        rating="unhelpful",
        comment="Needed more detail",
    )
    store.record_answer(
        "What is the refund limit?",
        grounded_answer(),
        top_k=4,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["feedback_id"] == first["feedback_id"]
    assert first["sources"] == [
        {"evidence_id": "E1", "filename": "refunds.txt", "chunk_index": 2}
    ]
    assert first["retrieved_chunk_ids"] == ["refund-chunk"]
    assert first["best_similarity"] == 0.8

    summary = store.summary()
    assert summary["total_answers"] == 3
    assert summary["helpful_count"] == 1
    assert summary["unhelpful_count"] == 1
    assert summary["rated_count"] == 2
    assert summary["helpful_rate"] == 0.5
    assert [item["comment"] for item in summary["latest_comments"]] == [
        "Needed more detail",
        "Correct and clear",
    ]


def test_empty_store_has_zero_summary(tmp_path):
    summary = FeedbackStore(tmp_path / "missing.jsonl").summary()

    assert summary == {
        "total_answers": 0,
        "helpful_count": 0,
        "unhelpful_count": 0,
        "rated_count": 0,
        "helpful_rate": None,
        "latest_comments": [],
    }


def test_invalid_rating_is_rejected(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    with pytest.raises(ValueError, match="rating must be"):
        store.record_answer(
            "Question",
            grounded_answer(),
            top_k=4,
            rating="maybe",
        )
