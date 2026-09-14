from app import print_grounded_answer
from src.models import GroundedAnswer, RetrievalResult, SourceReference


def result(text, source, chunk_index, similarity):
    return RetrievalResult(
        chunk_id=f"{source}-{chunk_index}",
        text=text,
        source=source,
        chunk_index=chunk_index,
        distance=1.0 - similarity,
    )


def test_grounded_output_maps_citation_to_source_and_excerpt(capsys):
    cited = result("Team Leads may approve up to $750.", "refunds.txt", 2, 0.81)
    uncited = result("Refund timing may vary.", "refunds.txt", 3, 0.62)
    answer = GroundedAnswer(
        answer="A Team Lead may approve up to $750. [E1]",
        confidence="High",
        decision_reason="Best retrieval similarity 0.810 met the threshold.",
        sources=(SourceReference("E1", "refunds.txt", 2),),
        evidence=(cited, uncited),
        cited_evidence_ids=("E1",),
        status="grounded",
    )

    print_grounded_answer(answer)
    output = capsys.readouterr().out

    assert "filename: refunds.txt; chunk index: 2" in output
    assert "Answer confidence: High" in output
    assert "Decision: Best retrieval similarity 0.810" in output
    assert "SUPPORTING EXCERPT:" in output
    assert "Team Leads may approve up to $750." in output
    assert "ADDITIONAL RETRIEVED EVIDENCE:" in output
    assert "[E2] refunds.txt · chunk index 3" in output
    assert "page:" not in output.lower()
    assert "section:" not in output.lower()


def test_insufficient_output_does_not_claim_a_supporting_source(capsys):
    retrieved = result("Some unrelated policy text.", "policy.txt", 4, 0.2)
    answer = GroundedAnswer(
        answer="I could not find enough information in the knowledge base.",
        confidence="Insufficient evidence",
        decision_reason="Best retrieval similarity 0.200 is below the threshold.",
        sources=(),
        evidence=(retrieved,),
        cited_evidence_ids=(),
        status="insufficient",
    )

    print_grounded_answer(answer)
    output = capsys.readouterr().out

    assert "No supporting source identified." in output
    assert "Answer confidence: Insufficient evidence" in output
    assert "None. The answer was not supported" in output
    assert "[E1] policy.txt · chunk index 4" in output
