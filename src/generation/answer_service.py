from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from src.config import (
    HIGH_CONFIDENCE_SIMILARITY,
    INSUFFICIENT_ANSWER,
    MIN_GROUNDING_SIMILARITY,
)
from src.generation.prompts import SYSTEM_INSTRUCTION, build_grounded_prompt
from src.generation.provider import GenerationProvider, GenerationUnavailableError
from src.models import GroundedAnswer, RetrievalResult, SourceReference
from src.retrieval.retriever import Retriever


CITATION_PATTERN = re.compile(r"\[E(\d+)\]")
NUMERIC_PATTERN = re.compile(r"(?<![A-Za-z])(?:[$₹£€]\s*)?\d[\d,]*(?:\.\d+)?%?")


def _numeric_tokens(text: str) -> Tuple[str, ...]:
    without_citations = CITATION_PATTERN.sub("", text)
    return tuple(
        dict.fromkeys(
            match.group(0).replace(" ", "").replace(",", "").lower()
            for match in NUMERIC_PATTERN.finditer(without_citations)
        )
    )


def _validate_citations(
    generated: str,
    evidence_by_id: Dict[str, RetrievalResult],
) -> Tuple[Tuple[str, ...], Optional[str], Tuple[str, ...]]:
    citation_numbers = CITATION_PATTERN.findall(generated)
    if not citation_numbers:
        return (), "The generated response contained no evidence citations and was rejected.", ()

    if any(number not in evidence_by_id for number in citation_numbers):
        return (
            (),
            "The generated response cited evidence that was not retrieved and was rejected.",
            (),
        )

    unique_numbers = tuple(dict.fromkeys(citation_numbers))
    answer_numbers = _numeric_tokens(generated)
    cited_numbers = {
        token
        for number in unique_numbers
        for token in _numeric_tokens(evidence_by_id[number].text)
    }
    missing_numbers = tuple(
        number for number in answer_numbers if number not in cited_numbers
    )
    if missing_numbers:
        return (
            unique_numbers,
            "The generated response cited evidence that did not contain its exact "
            "numeric claims and was rejected.",
            missing_numbers,
        )
    return unique_numbers, None, ()


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        provider: GenerationProvider,
        minimum_similarity: float = MIN_GROUNDING_SIMILARITY,
        high_confidence_similarity: float = HIGH_CONFIDENCE_SIMILARITY,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.minimum_similarity = minimum_similarity
        self.high_confidence_similarity = high_confidence_similarity

    def answer(self, question: str, top_k: int) -> GroundedAnswer:
        evidence = tuple(self.retriever.search(question, top_k=top_k))
        if not evidence:
            return self._insufficient(evidence, "No evidence chunks were retrieved.")

        best_similarity = evidence[0].similarity
        if best_similarity < self.minimum_similarity:
            return self._insufficient(
                evidence,
                f"Best retrieval similarity {best_similarity:.3f} is below the "
                f"minimum threshold {self.minimum_similarity:.2f}.",
            )

        prompt = build_grounded_prompt(question, evidence)
        try:
            generated = self.provider.generate(SYSTEM_INSTRUCTION, prompt).strip()
        except GenerationUnavailableError:
            return GroundedAnswer(
                answer=INSUFFICIENT_ANSWER,
                confidence="Insufficient evidence",
                decision_reason=(
                    "The local generation model was unavailable, so no answer "
                    "could be generated and validated."
                ),
                sources=(),
                evidence=evidence,
                cited_evidence_ids=(),
                status="generation_unavailable",
            )

        if INSUFFICIENT_ANSWER.lower() in generated.lower():
            return self._insufficient(
                evidence,
                "The local model reported that the retrieved context did not "
                "fully support an answer.",
            )

        evidence_by_id: Dict[str, RetrievalResult] = {
            str(index): result for index, result in enumerate(evidence, start=1)
        }
        unique_numbers, citation_error, missing_numbers = _validate_citations(
            generated, evidence_by_id
        )

        if missing_numbers:
            direct_evidence = tuple(
                f"[E{number}]"
                for number, result in evidence_by_id.items()
                if set(missing_numbers).issubset(set(_numeric_tokens(result.text)))
            )
            retry_prompt = (
                f"{prompt}\n\n"
                "A previous draft was rejected because its cited evidence did not "
                "contain these exact numeric claims: "
                f"{', '.join(missing_numbers)}.\n"
                f"Previous draft: {generated}\n"
                "Evidence blocks containing all of those exact values: "
                f"{', '.join(direct_evidence) if direct_evidence else 'none'}.\n"
                "Return a corrected answer with citations that directly support each "
                "claim. If no evidence block supports the answer, return the required "
                "insufficient-information response.\n\nCorrected answer:"
            )
            try:
                generated = self.provider.generate(
                    SYSTEM_INSTRUCTION, retry_prompt
                ).strip()
            except GenerationUnavailableError:
                return self._insufficient(
                    evidence,
                    "The local model became unavailable while correcting an "
                    "unsupported citation.",
                )

            if INSUFFICIENT_ANSWER.lower() in generated.lower():
                return self._insufficient(
                    evidence,
                    "The local model could not produce a directly supported citation.",
                )
            unique_numbers, citation_error, _ = _validate_citations(
                generated, evidence_by_id
            )

        if citation_error:
            return self._insufficient(
                evidence,
                citation_error,
            )

        sources = tuple(
            SourceReference(
                evidence_id=f"E{number}",
                filename=evidence_by_id[number].source,
                chunk_index=evidence_by_id[number].chunk_index,
            )
            for number in unique_numbers
        )

        return GroundedAnswer(
            answer=generated,
            confidence=(
                "High"
                if best_similarity >= self.high_confidence_similarity
                else "Medium"
            ),
            decision_reason=(
                f"Best retrieval similarity {best_similarity:.3f} met the "
                f"minimum threshold {self.minimum_similarity:.2f}; citations "
                f"{', '.join(f'E{number}' for number in unique_numbers)} were "
                "validated against retrieved chunks, including exact numeric claims."
            ),
            sources=sources,
            evidence=evidence,
            cited_evidence_ids=tuple(f"E{number}" for number in unique_numbers),
            status="grounded",
        )

    @staticmethod
    def _insufficient(
        evidence: Tuple[RetrievalResult, ...], decision_reason: str
    ) -> GroundedAnswer:
        return GroundedAnswer(
            answer=INSUFFICIENT_ANSWER,
            confidence="Insufficient evidence",
            decision_reason=decision_reason,
            sources=(),
            evidence=evidence,
            cited_evidence_ids=(),
            status="insufficient",
        )
