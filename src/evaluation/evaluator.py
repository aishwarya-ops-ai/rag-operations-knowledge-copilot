from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import RetrievalResult
from src.retrieval.retriever import Retriever
from src.config import PROJECT_ROOT


CSV_COLUMNS = {
    "question",
    "expected_source",
    "expected_answer_summary",
    "answerable_yes_no",
}


def _portable_project_path(path: Path) -> str:
    """Represent project files without recording a local machine path."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or len(data) < 10:
        raise ValueError("Evaluation file must contain at least 10 questions")
    return data


def load_evaluation_questions(path: Path) -> List[Dict[str, Any]]:
    """Load and validate the portfolio evaluation cases from CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = CSV_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Evaluation CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        questions: List[Dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            question = row["question"].strip()
            expected_source = row["expected_source"].strip()
            expected_summary = row["expected_answer_summary"].strip()
            answerable_label = row["answerable_yes_no"].strip().lower()
            if answerable_label not in {"yes", "no"}:
                raise ValueError(
                    f"Row {row_number} answerable_yes_no must be Yes or No"
                )
            if not question:
                raise ValueError(f"Row {row_number} has an empty question")

            questions.append(
                {
                    "id": f"q-{len(questions) + 1:02d}",
                    "question": question,
                    "expected_source": expected_source,
                    "expected_answer_summary": expected_summary,
                    "expected_answerable": answerable_label == "yes",
                }
            )

    if len(questions) < 10:
        raise ValueError("Evaluation CSV must contain at least 10 questions")
    if not any(item["expected_answerable"] for item in questions):
        raise ValueError("Evaluation CSV must include an answerable question")
    if not any(not item["expected_answerable"] for item in questions):
        raise ValueError("Evaluation CSV must include an unanswerable question")
    return questions


def _normalized_source(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"^\d+[\s_-]*", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def _expected_source_rank(
    expected_source: str,
    results: List[RetrievalResult],
) -> Optional[int]:
    expected = _normalized_source(expected_source)
    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if _normalized_source(result.source) == expected
        ),
        None,
    )


def evaluate_portfolio_rag(
    retriever: Retriever,
    questions_path: Path,
    top_k: int,
    minimum_similarity: float,
) -> Dict[str, Any]:
    """Run a small, deterministic portfolio evaluation without calling an LLM."""
    questions = load_evaluation_questions(questions_path)
    cases: List[Dict[str, Any]] = []

    for item in questions:
        results: List[RetrievalResult] = retriever.search(item["question"], top_k)
        top_similarity = results[0].similarity if results else None
        predicted_answerable = bool(
            top_similarity is not None and top_similarity >= minimum_similarity
        )
        expected_answerable = item["expected_answerable"]
        expected_source_rank = (
            _expected_source_rank(item["expected_source"], results)
            if expected_answerable
            else None
        )
        retrieved_sources = list(dict.fromkeys(result.source for result in results))

        cases.append(
            {
                **item,
                "retrieved_sources": retrieved_sources,
                "retrieved_results": [
                    {
                        "source": result.source,
                        "chunk_index": result.chunk_index,
                        "similarity": round(result.similarity, 4),
                    }
                    for result in results
                ],
                "top_similarity": (
                    round(top_similarity, 4) if top_similarity is not None else None
                ),
                "predicted_answerable": predicted_answerable,
                "answerability_correct": predicted_answerable == expected_answerable,
                "expected_source_rank": expected_source_rank,
                "retrieval_hit": expected_source_rank is not None,
                "top_source_match": expected_source_rank == 1,
                "unsupported_answer": (
                    not expected_answerable and predicted_answerable
                ),
            }
        )

    answerable_cases = [case for case in cases if case["expected_answerable"]]
    unanswerable_cases = [case for case in cases if not case["expected_answerable"]]
    retrieval_hits = sum(case["retrieval_hit"] for case in answerable_cases)
    source_matches = sum(case["top_source_match"] for case in answerable_cases)
    classification_correct = sum(case["answerability_correct"] for case in cases)
    unsupported_answers = sum(case["unsupported_answer"] for case in cases)

    return {
        "framework": "portfolio_rag_evaluation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": _portable_project_path(questions_path),
        "top_k": top_k,
        "minimum_similarity": minimum_similarity,
        "question_count": len(cases),
        "answerable_count": len(answerable_cases),
        "unanswerable_count": len(unanswerable_cases),
        "retrieval_hits": retrieval_hits,
        "retrieval_hit_rate": retrieval_hits / len(answerable_cases),
        "answerability_correct": classification_correct,
        "answerability_classification_accuracy": classification_correct / len(cases),
        "unsupported_answers": unsupported_answers,
        "unsupported_answer_rate": unsupported_answers / len(unanswerable_cases),
        "source_matches": source_matches,
        "source_match_rate": source_matches / len(answerable_cases),
        "cases": cases,
    }


def evaluate_retrieval(
    retriever: Retriever,
    questions_path: Path,
    top_k: int,
) -> Dict[str, Any]:
    questions = load_questions(questions_path)
    cases: List[Dict[str, Any]] = []

    for item in questions:
        results: List[RetrievalResult] = retriever.search(item["question"], top_k)
        expected_source = item["expected_source"]
        expected_terms = [term.lower() for term in item.get("expected_terms", [])]
        matched_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if result.source == expected_source
                and all(term in result.text.lower() for term in expected_terms)
            ),
            None,
        )
        matched = matched_rank is not None
        cases.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_source": expected_source,
                "expected_terms": item.get("expected_terms", []),
                "matched": matched,
                "rank": matched_rank,
                "retrieved_sources": [result.source for result in results],
                "top_similarity": round(results[0].similarity, 4) if results else None,
            }
        )

    hits = sum(1 for case in cases if case["matched"])
    reciprocal_rank = sum(
        1.0 / case["rank"] for case in cases if case["rank"] is not None
    ) / len(cases)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "question_count": len(cases),
        "hits": hits,
        "hit_rate": hits / len(cases),
        "mean_reciprocal_rank": reciprocal_rank,
        "cases": cases,
    }


def append_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
