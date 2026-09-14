import json
import csv

import pytest

from src.config import EVALUATION_FILE, PARAPHRASE_EVALUATION_FILE
from src.evaluation.evaluator import (
    evaluate_portfolio_rag,
    evaluate_retrieval,
    load_evaluation_questions,
    load_paraphrase_questions,
    load_questions,
)
from src.models import RetrievalResult


class FakeRetriever:
    def search(self, query, top_k):
        return [
            RetrievalResult(
                chunk_id="id",
                text="evidence",
                source="expected.txt",
                chunk_index=0,
                distance=0.1,
            )
        ]


def test_evaluation_requires_at_least_ten_questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10"):
        load_questions(path)


def test_evaluation_calculates_hit_rate(tmp_path):
    path = tmp_path / "questions.json"
    questions = [
        {"id": f"q-{index}", "question": f"Question {index}", "expected_source": "expected.txt"}
        for index in range(10)
    ]
    path.write_text(json.dumps(questions), encoding="utf-8")

    result = evaluate_retrieval(FakeRetriever(), path, top_k=4)

    assert result["question_count"] == 10
    assert result["hit_rate"] == 1.0
    assert result["mean_reciprocal_rank"] == 1.0


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question",
                "expected_source",
                "expected_answer_summary",
                "answerable_yes_no",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_paraphrase_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_id",
                "paraphrase_id",
                "question",
                "expected_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


class PortfolioRetriever:
    def search(self, query, top_k):
        if query == "unsupported false positive":
            return [
                RetrievalResult(
                    chunk_id="fp",
                    text="related but insufficient",
                    source="02_Customer_Escalation_Policy.txt",
                    chunk_index=0,
                    distance=0.3,
                )
            ]
        if query.startswith("unsupported"):
            return [
                RetrievalResult(
                    chunk_id="safe",
                    text="unrelated",
                    source="02_Customer_Escalation_Policy.txt",
                    chunk_index=0,
                    distance=0.8,
                )
            ]
        return [
            RetrievalResult(
                chunk_id="right",
                text="relevant evidence",
                source="01_Attendance_Shift_CheckIn_SOP.txt",
                chunk_index=1,
                distance=0.2,
            ),
            RetrievalResult(
                chunk_id="other",
                text="other evidence",
                source="02_Customer_Escalation_Policy.txt",
                chunk_index=0,
                distance=0.4,
            ),
        ][:top_k]


def test_portfolio_evaluation_reports_requested_metrics(tmp_path):
    path = tmp_path / "evaluation_questions.csv"
    rows = [
        {
            "question": f"answerable {index}",
            "expected_source": "Attendance & Shift Check-In SOP",
            "expected_answer_summary": "Covered by the attendance SOP.",
            "answerable_yes_no": "Yes",
        }
        for index in range(8)
    ]
    rows.extend(
        [
            {
                "question": "unsupported safe",
                "expected_source": "None — not covered",
                "expected_answer_summary": "Not covered.",
                "answerable_yes_no": "No",
            },
            {
                "question": "unsupported false positive",
                "expected_source": "None — not covered",
                "expected_answer_summary": "Not covered.",
                "answerable_yes_no": "No",
            },
        ]
    )
    write_csv(path, rows)

    result = evaluate_portfolio_rag(
        PortfolioRetriever(), path, top_k=2, minimum_similarity=0.5
    )

    assert result["framework"] == "portfolio_rag_evaluation"
    assert result["dataset"] == "evaluation_questions.csv"
    assert result["retrieval_hit_rate"] == 1.0
    assert result["source_match_rate"] == 1.0
    assert result["answerability_classification_accuracy"] == 0.9
    assert result["unsupported_answer_rate"] == 0.5
    assert result["cases"][-1]["unsupported_answer"] is True
    assert result["cases"][-1]["retrieved_sources"] == [
        "02_Customer_Escalation_Policy.txt"
    ]


def test_portfolio_csv_rejects_invalid_answerability_label(tmp_path):
    path = tmp_path / "evaluation_questions.csv"
    write_csv(
        path,
        [
            {
                "question": f"Question {index}",
                "expected_source": "policy.txt",
                "expected_answer_summary": "Summary",
                "answerable_yes_no": "Maybe",
            }
            for index in range(10)
        ],
    )

    with pytest.raises(ValueError, match="must be Yes or No"):
        load_evaluation_questions(path)


def test_paraphrase_evaluation_reports_robustness_metrics(tmp_path):
    canonical_path = tmp_path / "evaluation_questions.csv"
    canonical_rows = [
        {
            "question": f"answerable {index}",
            "expected_source": "Attendance & Shift Check-In SOP",
            "expected_answer_summary": "Covered by the attendance SOP.",
            "answerable_yes_no": "Yes",
        }
        for index in range(10)
    ]
    canonical_rows.append(
        {
            "question": "unsupported safe",
            "expected_source": "None — not covered",
            "expected_answer_summary": "Not covered.",
            "answerable_yes_no": "No",
        }
    )
    write_csv(canonical_path, canonical_rows)

    paraphrase_path = tmp_path / "evaluation_paraphrases.csv"
    paraphrase_rows = [
        {
            "canonical_id": f"q-{canonical_index:02d}",
            "paraphrase_id": f"q-{canonical_index:02d}-p{variant}",
            "question": (
                "unsupported paraphrase failure"
                if canonical_index == 10 and variant == 3
                else f"answerable paraphrase {canonical_index}-{variant}"
            ),
            "expected_source": "Attendance & Shift Check-In SOP",
        }
        for canonical_index in range(1, 11)
        for variant in range(1, 4)
    ]
    write_paraphrase_csv(paraphrase_path, paraphrase_rows)

    result = evaluate_portfolio_rag(
        PortfolioRetriever(),
        canonical_path,
        top_k=2,
        minimum_similarity=0.5,
        paraphrases_path=paraphrase_path,
    )

    assert result["canonical_query_accuracy"] == 1.0
    assert result["paraphrase_count"] == 30
    assert result["paraphrase_successes"] == 29
    assert result["paraphrase_retrieval_success_rate"] == pytest.approx(29 / 30)
    assert result["paraphrase_accuracy"] == pytest.approx(29 / 30)
    assert result["unsupported_question_refusal_accuracy"] == 1.0
    assert result["paraphrase_cases"][-1]["paraphrase_pass"] is False


def test_paraphrase_file_requires_three_variants_per_canonical_question(tmp_path):
    canonical_path = tmp_path / "evaluation_questions.csv"
    rows = [
        {
            "question": f"answerable {index}",
            "expected_source": "Attendance & Shift Check-In SOP",
            "expected_answer_summary": "Covered.",
            "answerable_yes_no": "Yes",
        }
        for index in range(10)
    ]
    rows.append(
        {
            "question": "unsupported safe",
            "expected_source": "None — not covered",
            "expected_answer_summary": "Not covered.",
            "answerable_yes_no": "No",
        }
    )
    write_csv(canonical_path, rows)
    canonical = load_evaluation_questions(canonical_path)

    paraphrase_path = tmp_path / "evaluation_paraphrases.csv"
    paraphrases = [
        {
            "canonical_id": f"q-{canonical_index:02d}",
            "paraphrase_id": f"q-{canonical_index:02d}-p{variant}",
            "question": f"variant {canonical_index}-{variant}",
            "expected_source": "Attendance & Shift Check-In SOP",
        }
        for canonical_index in range(1, 11)
        for variant in range(1, 4)
        if not (canonical_index == 10 and variant == 3)
    ]
    write_paraphrase_csv(paraphrase_path, paraphrases)

    with pytest.raises(ValueError, match="exactly 3 paraphrases"):
        load_paraphrase_questions(paraphrase_path, canonical)


def test_project_paraphrase_dataset_has_ten_groups_of_three():
    canonical = load_evaluation_questions(EVALUATION_FILE)
    paraphrases = load_paraphrase_questions(
        PARAPHRASE_EVALUATION_FILE,
        canonical,
    )

    counts = {}
    for item in paraphrases:
        counts[item["canonical_id"]] = counts.get(item["canonical_id"], 0) + 1

    assert len(paraphrases) == 30
    assert len(counts) == 10
    assert set(counts.values()) == {3}
