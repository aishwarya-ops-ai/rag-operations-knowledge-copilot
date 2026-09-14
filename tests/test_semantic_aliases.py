from src.ingestion.semantic_aliases import (
    add_retrieval_aliases,
    alias_query_variants,
    aliases_for_text,
)
from src.models import Chunk


def test_shift_check_in_aliases_cover_common_operational_wording():
    aliases = aliases_for_text(
        "When the employee forgot to check in, a retroactive check-in may be approved."
    )

    assert "clock in" in aliases
    assert "missed check-in" in aliases
    assert "forgot to clock in" in aliases
    assert "failed to check in" in aliases


def test_other_supported_operational_concepts_receive_narrow_aliases():
    assert "money back" in aliases_for_text("The customer may request a refund.")
    assert "QA review" in aliases_for_text("The quality review checks accuracy.")
    assert "new hire onboarding" in aliases_for_text(
        "A trainee completes the standard program."
    )
    assert "speak to a supervisor" in aliases_for_text(
        "A customer asking for a supervisor may require escalation."
    )


def test_unrelated_text_does_not_receive_aliases():
    assert aliases_for_text("The office kitchen closes at six.") == ()


def test_aliases_are_separate_from_citation_text():
    original = "Employees must complete the required check-in procedure."
    chunk = Chunk(
        id="chunk-1",
        text=original,
        source="attendance.txt",
        chunk_index=0,
        character_count=len(original),
    )

    enriched = add_retrieval_aliases([chunk])[0]

    assert enriched.text == original
    assert "clock in" in enriched.retrieval_aliases


def test_known_aliases_create_canonical_query_variants():
    assert alias_query_variants("How do I clock in?") == [
        "employee shift check-in attendance procedure"
    ]
    assert alias_query_variants("Can I speak to a supervisor?") == [
        "customer requests supervisor escalation policy"
    ]
    assert alias_query_variants("Show me the QA review process") == [
        "employee quality review guidelines"
    ]


def test_ambiguous_or_unknown_wording_does_not_trigger_aliases():
    assert alias_query_variants("I want to speak to someone") == []
    assert alias_query_variants("Who is the CEO?") == []
