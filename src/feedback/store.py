from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.config import FEEDBACK_FILE
from src.models import GroundedAnswer


VALID_RATINGS = {"helpful", "unhelpful"}


class FeedbackStore:
    """Append-only JSONL storage for local, single-user feedback."""

    def __init__(self, path: Path = FEEDBACK_FILE) -> None:
        self.path = path

    def record_answer(
        self,
        question: str,
        result: GroundedAnswer,
        top_k: int,
        rating: Optional[str] = None,
        comment: str = "",
    ) -> Dict[str, Any]:
        if rating is not None and rating not in VALID_RATINGS:
            raise ValueError("rating must be 'helpful', 'unhelpful', or None")

        record = {
            "feedback_id": str(uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": result.answer,
            "answer_status": result.status,
            "answer_confidence": result.confidence,
            "decision_reason": result.decision_reason,
            "rating": rating,
            "comment": comment.strip(),
            "top_k": top_k,
            "best_similarity": (
                round(result.evidence[0].similarity, 4) if result.evidence else None
            ),
            "sources": [
                {
                    "evidence_id": source.evidence_id,
                    "filename": source.filename,
                    "chunk_index": source.chunk_index,
                }
                for source in result.sources
            ],
            "retrieved_chunk_ids": [item.chunk_id for item in result.evidence],
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def records(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def summary(self, latest_comment_limit: int = 5) -> Dict[str, Any]:
        records = self.records()
        helpful = sum(record.get("rating") == "helpful" for record in records)
        unhelpful = sum(record.get("rating") == "unhelpful" for record in records)
        rated = helpful + unhelpful
        comments = [
            {
                "timestamp_utc": record.get("timestamp_utc", ""),
                "rating": record.get("rating"),
                "comment": str(record.get("comment", "")),
            }
            for record in reversed(records)
            if str(record.get("comment", "")).strip()
        ][:latest_comment_limit]
        return {
            "total_answers": len(records),
            "helpful_count": helpful,
            "unhelpful_count": unhelpful,
            "rated_count": rated,
            "helpful_rate": helpful / rated if rated else None,
            "latest_comments": comments,
        }
