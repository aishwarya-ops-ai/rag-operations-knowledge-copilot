from __future__ import annotations

import argparse
import sys
from typing import Iterable

from src.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    EVALUATION_FILE,
    EVALUATION_RESULTS_FILE,
    MIN_GROUNDING_SIMILARITY,
    PARAPHRASE_EVALUATION_FILE,
)
from src.evaluation.evaluator import append_result, evaluate_portfolio_rag
from src.feedback.store import FeedbackStore
from src.generation.answer_service import AnswerService
from src.generation.provider import OllamaProvider
from src.ingestion.pipeline import prepare_chunks
from src.models import GroundedAnswer, RetrievalResult
from src.retrieval.embeddings import LocalEmbedder
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import ChromaVectorStore


def build_services(allow_model_download: bool = False):
    embedder = LocalEmbedder(allow_download=allow_model_download)
    vector_store = ChromaVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    return embedder, vector_store, retriever


def print_results(results: Iterable[RetrievalResult]) -> None:
    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] {result.source} · chunk {result.chunk_index}")
        print(f"Cosine similarity: {result.similarity:.4f}")
        print("-" * 72)
        print(result.text)


def print_grounded_answer(result: GroundedAnswer) -> None:
    print("\nFINAL ANSWER")
    print("-" * 72)
    print(result.answer)
    print(f"\nAnswer confidence: {result.confidence}")
    print(f"Decision: {result.decision_reason}")

    print("\nSOURCE:")
    print("-" * 72)
    if result.sources:
        for source in result.sources:
            print(
                f"- [{source.evidence_id}] filename: {source.filename}; "
                f"chunk index: {source.chunk_index}"
            )
    else:
        print("- No supporting source identified.")

    print("\nSUPPORTING EXCERPT:")
    print("-" * 72)
    if result.sources:
        evidence_by_id = {
            f"E{index}": evidence
            for index, evidence in enumerate(result.evidence, start=1)
        }
        for source in result.sources:
            evidence = evidence_by_id[source.evidence_id]
            print(
                f"\n[{source.evidence_id}] {source.filename} · "
                f"chunk index {source.chunk_index}"
            )
            print(evidence.text)
    else:
        print("- None. The answer was not supported by retrieved evidence.")

    cited_ids = {source.evidence_id for source in result.sources}
    additional_evidence = [
        (f"E{index}", evidence)
        for index, evidence in enumerate(result.evidence, start=1)
        if f"E{index}" not in cited_ids
    ]
    if additional_evidence:
        print("\nADDITIONAL RETRIEVED EVIDENCE:")
        print("-" * 72)
        for evidence_id, evidence in additional_evidence:
            print(
                f"\n[{evidence_id}] {evidence.source} · "
                f"chunk index {evidence.chunk_index}"
            )
            print(f"Cosine similarity: {evidence.similarity:.4f}")
            print(evidence.text)


def print_feedback_summary(store: FeedbackStore) -> None:
    summary = store.summary()
    helpful_rate = (
        f"{summary['helpful_rate']:.1%}"
        if summary["helpful_rate"] is not None
        else "Not available"
    )
    print("\nFEEDBACK SUMMARY")
    print("-" * 72)
    print(f"Total answers: {summary['total_answers']}")
    print(f"Helpful: {summary['helpful_count']}")
    print(f"Unhelpful: {summary['unhelpful_count']}")
    print(f"Helpful rate: {helpful_rate} (among rated answers)")
    print("Latest comments:")
    if summary["latest_comments"]:
        for item in summary["latest_comments"]:
            rating = item["rating"] or "unrated"
            print(f"- [{rating}] {item['comment']}")
    else:
        print("- None")


def collect_feedback(
    store: FeedbackStore,
    question: str,
    result: GroundedAnswer,
    top_k: int,
) -> None:
    print("\nFEEDBACK")
    print("-" * 72)
    print("Helpful 👍  /  Not helpful 👎  /  Skip")

    rating = None
    comment = ""
    if sys.stdin.isatty():
        try:
            choice = input("Choose [h/n/s]: ").strip().lower()
            rating = {
                "h": "helpful",
                "helpful": "helpful",
                "👍": "helpful",
                "n": "unhelpful",
                "not helpful": "unhelpful",
                "unhelpful": "unhelpful",
                "👎": "unhelpful",
            }.get(choice)
            if choice and choice not in {"s", "skip"} and rating is None:
                print("Unrecognized choice; saving this answer as unrated.")
            comment = input("Optional comment: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFeedback skipped.")
    else:
        print("Interactive rating skipped because input is not a terminal.")

    store.record_answer(
        question=question,
        result=result,
        top_k=top_k,
        rating=rating,
        comment=comment,
    )
    print_feedback_summary(store)


def index_command() -> None:
    embedder, vector_store, _ = build_services(allow_model_download=True)
    chunks = prepare_chunks()
    print(f"Loaded {len(chunks)} chunks. Generating local embeddings...")
    embeddings = embedder.encode(chunk.text for chunk in chunks)
    vector_store.replace(chunks, embeddings)
    source_count = len({chunk.source for chunk in chunks})
    print(
        f"Indexed {len(chunks)} chunks from {source_count} documents "
        f"using {EMBEDDING_MODEL}."
    )


def query_command(query: str, top_k: int) -> None:
    _, _, retriever = build_services()
    print_results(retriever.search(query, top_k=top_k))


def answer_command(query: str, top_k: int) -> None:
    _, _, retriever = build_services()
    service = AnswerService(retriever=retriever, provider=OllamaProvider())
    result = service.answer(query, top_k=top_k)
    print_grounded_answer(result)
    collect_feedback(FeedbackStore(), query, result, top_k)


def chat_command(top_k: int) -> None:
    _, vector_store, retriever = build_services()
    service = AnswerService(retriever=retriever, provider=OllamaProvider())
    print(f"Operations Knowledge Copilot — Grounded Q&A ({vector_store.count()} chunks)")
    print("Enter a question, or type 'quit' to exit.")
    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"quit", "exit", ":q"}:
            return
        if not query:
            continue
        result = service.answer(query, top_k=top_k)
        print_grounded_answer(result)
        collect_feedback(FeedbackStore(), query, result, top_k)


def interactive_command(top_k: int) -> None:
    _, vector_store, retriever = build_services()
    print(f"Operations Knowledge Copilot — Retrieval CLI ({vector_store.count()} chunks)")
    print("Enter a question, or type 'quit' to exit.")
    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"quit", "exit", ":q"}:
            return
        if not query:
            continue
        print_results(retriever.search(query, top_k=top_k))


def evaluate_command(top_k: int) -> None:
    _, _, retriever = build_services()
    result = evaluate_portfolio_rag(
        retriever,
        EVALUATION_FILE,
        top_k=top_k,
        minimum_similarity=MIN_GROUNDING_SIMILARITY,
        paraphrases_path=PARAPHRASE_EVALUATION_FILE,
    )
    append_result(EVALUATION_RESULTS_FILE, result)

    print("\nPORTFOLIO RAG EVALUATION FRAMEWORK")
    print("=" * 72)
    print("This is a small project diagnostic, not a formal production benchmark.")
    print(
        f"Cases: {result['question_count']} "
        f"({result['answerable_count']} answerable, "
        f"{result['unanswerable_count']} unanswerable)"
    )
    print(
        f"Configuration: top-k {result['top_k']}; answerability similarity "
        f"threshold {result['minimum_similarity']:.2f}"
    )

    print("\nCASE RESULTS")
    print("-" * 72)
    for case in result["cases"]:
        marker = "PASS" if case["answerability_correct"] else "FAIL"
        expected = "answerable" if case["expected_answerable"] else "unanswerable"
        predicted = "answerable" if case["predicted_answerable"] else "unanswerable"
        rank = case["expected_source_rank"] or "—"
        similarity = (
            f"{case['top_similarity']:.3f}"
            if case["top_similarity"] is not None
            else "—"
        )
        print(
            f"{case['id']} | Answerability {marker} | expected {expected}; "
            f"predicted {predicted}; top similarity {similarity}"
        )
        print(f"    {case['question']}")
        if case["expected_answerable"]:
            print(
                f"    Expected source: {case['expected_source']} "
                f"(top-k rank {rank})"
            )
        else:
            print("    Expected source: none; information is not covered")
        retrieved = ", ".join(case["retrieved_sources"]) or "None"
        print(f"    Retrieved sources: {retrieved}")

    print("\nEVALUATION SUMMARY")
    print("-" * 72)
    print(
        f"Retrieval hit rate: {result['retrieval_hits']}/"
        f"{result['answerable_count']} ({result['retrieval_hit_rate']:.1%})"
    )
    print(
        f"Answerability classification accuracy: "
        f"{result['answerability_correct']}/{result['question_count']} "
        f"({result['answerability_classification_accuracy']:.1%})"
    )
    print(
        f"Unsupported-answer rate: {result['unsupported_answers']}/"
        f"{result['unanswerable_count']} ({result['unsupported_answer_rate']:.1%})"
    )
    print(
        f"Source-match rate: {result['source_matches']}/"
        f"{result['answerable_count']} ({result['source_match_rate']:.1%})"
    )
    print(
        f"Canonical query accuracy: {result['canonical_query_successes']}/"
        f"{result['canonical_query_count']} "
        f"({result['canonical_query_accuracy']:.1%})"
    )
    print(
        f"Paraphrase Retrieval Success Rate: "
        f"{result['paraphrase_successes']}/{result['paraphrase_count']} "
        f"({result['paraphrase_retrieval_success_rate']:.1%})"
    )
    print(
        f"Unsupported-question refusal accuracy: "
        f"{result['unsupported_refusals']}/{result['unanswerable_count']} "
        f"({result['unsupported_question_refusal_accuracy']:.1%})"
    )

    failed_paraphrases = [
        case for case in result["paraphrase_cases"] if not case["paraphrase_pass"]
    ]
    if failed_paraphrases:
        print("\nPARAPHRASE FAILURES")
        print("-" * 72)
        for case in failed_paraphrases:
            rank = case["expected_source_rank"] or "—"
            similarity = (
                f"{case['top_similarity']:.3f}"
                if case["top_similarity"] is not None
                else "—"
            )
            print(
                f"{case['id']} | expected-source rank {rank}; "
                f"top similarity {similarity}"
            )
            print(f"    {case['question']}")
    print("\nMetric definitions:")
    print(f"- Retrieval hit: expected source appears anywhere in the top {top_k}.")
    print("- Source match: expected source is the first-ranked result.")
    print(
        "- Answerability: top similarity meets the same threshold used before "
        "answer generation."
    )
    print(
        "- Unsupported answer: an expected-unanswerable question crosses that "
        "retrieval gate. It measures risk, not verified model hallucination."
    )
    print(
        "- Paraphrase success: expected source appears in top-k and the best "
        "similarity passes the pre-generation evidence threshold."
    )
    print("- This evaluation does not grade the generated answer prose.")
    print(f"Result appended to {EVALUATION_RESULTS_FILE}")


def feedback_command() -> None:
    print_feedback_summary(FeedbackStore())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local document retrieval and grounded answer generation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Load, chunk, embed, and index all TXT files")

    query_parser = subparsers.add_parser("query", help="Run one retrieval query")
    query_parser.add_argument("query", help="Natural-language question")
    query_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    answer_parser = subparsers.add_parser(
        "answer", help="Retrieve evidence and generate a grounded local answer"
    )
    answer_parser.add_argument("query", help="Natural-language question")
    answer_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    interactive_parser = subparsers.add_parser(
        "interactive", help="Open a small interactive retrieval prompt"
    )
    interactive_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    chat_parser = subparsers.add_parser(
        "chat", help="Open an interactive grounded question-answering prompt"
    )
    chat_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run the portfolio RAG evaluation framework"
    )
    evaluate_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    subparsers.add_parser("feedback", help="Show the local answer-feedback summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "index":
            index_command()
        elif args.command == "query":
            query_command(args.query, args.top_k)
        elif args.command == "answer":
            answer_command(args.query, args.top_k)
        elif args.command == "interactive":
            interactive_command(args.top_k)
        elif args.command == "chat":
            chat_command(args.top_k)
        elif args.command == "evaluate":
            evaluate_command(args.top_k)
        elif args.command == "feedback":
            feedback_command()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
