"""Browser interface for the local Operations Knowledge Copilot."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st

from app import build_services
from src.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    EVALUATION_FILE,
    EVALUATION_RESULTS_FILE,
    HIGH_CONFIDENCE_SIMILARITY,
    MIN_GROUNDING_SIMILARITY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.evaluation.evaluator import append_result, evaluate_portfolio_rag
from src.feedback.store import FeedbackStore
from src.generation.answer_service import AnswerService
from src.generation.provider import OllamaProvider
from src.models import GroundedAnswer
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import ChromaVectorStore


st.set_page_config(
    page_title="Operations Knowledge Copilot",
    page_icon="📘",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the local knowledge base...")
def load_services() -> tuple[ChromaVectorStore, Retriever, AnswerService, FeedbackStore]:
    """Build and cache the same local services used by the CLI."""
    _, vector_store, retriever = build_services()
    answer_service = AnswerService(retriever=retriever, provider=OllamaProvider())
    return vector_store, retriever, answer_service, FeedbackStore()


def ollama_is_available() -> bool:
    """Check whether the configured local Ollama server is responding."""
    request = Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=1):
            return True
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def render_confidence(result: GroundedAnswer) -> None:
    label = f"Answer confidence: {result.confidence}"
    if result.confidence == "High":
        st.success(label)
    elif result.confidence == "Medium":
        st.warning(label)
    else:
        st.error(label)
    st.caption(result.decision_reason)


def render_sources_and_evidence(result: GroundedAnswer) -> None:
    st.markdown("#### Source references")
    if result.sources:
        for source in result.sources:
            st.markdown(
                f"- **[{source.evidence_id}] {source.filename}** — "
                f"chunk {source.chunk_index}"
            )
    else:
        st.write("No source was cited because the available evidence was insufficient.")

    st.markdown("#### Supporting excerpts")
    if result.sources:
        evidence_by_id = {
            f"E{index}": evidence
            for index, evidence in enumerate(result.evidence, start=1)
        }
        for source in result.sources:
            evidence = evidence_by_id.get(source.evidence_id)
            if evidence is None:
                continue
            st.markdown(
                f"**[{source.evidence_id}] {source.filename}, "
                f"chunk {source.chunk_index}**"
            )
            st.info(evidence.text)
    else:
        st.write("No excerpt directly supported an answer.")

    with st.expander("Show retrieved evidence"):
        if not result.evidence:
            st.write("No chunks were retrieved.")
        for index, evidence in enumerate(result.evidence, start=1):
            st.markdown(
                f"**[E{index}] {evidence.source}, chunk {evidence.chunk_index} "
                f"— similarity {evidence.similarity:.3f}**"
            )
            st.write(evidence.text)
            st.divider()


def render_feedback(
    question: str,
    result: GroundedAnswer,
    top_k: int,
    response_id: str,
    feedback_store: FeedbackStore,
) -> None:
    submitted = st.session_state.feedback_submitted.get(response_id)
    if submitted:
        st.success(f"Feedback saved locally: {submitted}.")
        return

    comment = st.text_input(
        "Optional feedback comment",
        key=f"comment_{response_id}",
        placeholder="What was useful, missing, or incorrect?",
    )
    helpful_col, unhelpful_col, _ = st.columns([1, 1, 4])
    if helpful_col.button("Helpful 👍", key=f"helpful_{response_id}"):
        feedback_store.record_answer(
            question=question,
            result=result,
            top_k=top_k,
            rating="helpful",
            comment=comment,
        )
        st.session_state.feedback_submitted[response_id] = "Helpful"
        st.rerun()
    if unhelpful_col.button("Not helpful 👎", key=f"unhelpful_{response_id}"):
        feedback_store.record_answer(
            question=question,
            result=result,
            top_k=top_k,
            rating="unhelpful",
            comment=comment,
        )
        st.session_state.feedback_submitted[response_id] = "Not helpful"
        st.rerun()


def render_answer(
    question: str,
    result: GroundedAnswer,
    top_k: int,
    response_id: str,
    feedback_store: FeedbackStore,
) -> None:
    st.markdown(result.answer)
    render_confidence(result)
    render_sources_and_evidence(result)
    st.markdown("#### Was this answer helpful?")
    render_feedback(question, result, top_k, response_id, feedback_store)


def render_feedback_summary(feedback_store: FeedbackStore) -> None:
    summary = feedback_store.summary()
    st.markdown("### Feedback summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total answers", summary["total_answers"])
    col2.metric("Helpful", summary["helpful_count"])
    col3.metric("Not helpful", summary["unhelpful_count"])
    helpful_rate = summary["helpful_rate"]
    col4.metric(
        "Helpful rate",
        f"{helpful_rate:.1%}" if helpful_rate is not None else "N/A",
    )

    if summary["latest_comments"]:
        with st.expander("Latest comments"):
            for item in summary["latest_comments"]:
                rating = item["rating"] or "unrated"
                st.markdown(f"- **{rating}** — {item['comment']}")


def render_evaluation(retriever: Retriever, top_k: int) -> None:
    st.subheader("Portfolio evaluation framework")
    st.write(
        "Run the local evaluation dataset to check retrieval and answerability "
        "behavior. This is a compact portfolio diagnostic, not a production benchmark."
    )

    if st.button("Run evaluation dataset", type="primary"):
        with st.spinner("Running retrieval evaluation..."):
            result = evaluate_portfolio_rag(
                retriever=retriever,
                questions_path=EVALUATION_FILE,
                top_k=top_k,
                minimum_similarity=MIN_GROUNDING_SIMILARITY,
            )
            append_result(EVALUATION_RESULTS_FILE, result)
            st.session_state.evaluation_result = result

    result = st.session_state.get("evaluation_result")
    if not result:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retrieval hit rate", f"{result['retrieval_hit_rate']:.1%}")
    col2.metric("Source match rate", f"{result['source_match_rate']:.1%}")
    col3.metric(
        "Unsupported-answer rate", f"{result['unsupported_answer_rate']:.1%}"
    )
    col4.metric(
        "Answerability accuracy",
        f"{result['answerability_classification_accuracy']:.1%}",
    )
    st.caption(
        f"Evaluated {result['question_count']} questions with top-k={result['top_k']} "
        f"and similarity threshold {result['minimum_similarity']:.2f}. Results were "
        "saved to the ignored local evaluation history."
    )

    table_rows = [
        {
            "Question": case["question"],
            "Expected": "Answerable" if case["expected_answerable"] else "Unanswerable",
            "Predicted": "Answerable" if case["predicted_answerable"] else "Unanswerable",
            "Expected source": case["expected_source"] or "None",
            "Retrieved sources": ", ".join(case["retrieved_sources"]),
            "Top similarity": case["top_similarity"],
        }
        for case in result["cases"]
    ]
    with st.expander("Show question-level results"):
        st.dataframe(table_rows, width="stretch", hide_index=True)


vector_store, retriever, answer_service, feedback_store = load_services()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = {}

with st.sidebar:
    st.header("Local system status")
    st.metric("Indexed documents", vector_store.source_count())
    st.metric("Indexed chunks", vector_store.count())
    st.caption("Embedding model")
    st.code(EMBEDDING_MODEL, language=None)
    top_k = st.slider(
        "Retrieval top-k", min_value=1, max_value=8, value=DEFAULT_TOP_K
    )
    st.caption("Local answer model")
    st.code(OLLAMA_MODEL, language=None)
    if ollama_is_available():
        st.success("Ollama is running")
    else:
        st.warning(
            "Ollama is not responding. Retrieval still works, but answer generation "
            "will use the safe fallback."
        )

st.title("Operations Knowledge Copilot")
st.write("Ask questions about SOPs, policies and training documents.")

ask_tab, evaluation_tab = st.tabs(["Ask the copilot", "Evaluation"])

with ask_tab:
    index_is_empty = vector_store.count() == 0
    if index_is_empty:
        st.warning("The vector index is empty. Run the existing CLI index command first.")

    for message in st.session_state.messages:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            render_answer(
                question=message["question"],
                result=message["result"],
                top_k=message["top_k"],
                response_id=message["response_id"],
                feedback_store=feedback_store,
            )

    question = st.chat_input("Ask an operations question", disabled=index_is_empty)
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base and grounding the answer..."):
                result = answer_service.answer(question, top_k=top_k)
            response_id = uuid4().hex
            st.session_state.messages.append(
                {
                    "question": question,
                    "result": result,
                    "top_k": top_k,
                    "response_id": response_id,
                }
            )
            render_answer(
                question=question,
                result=result,
                top_k=top_k,
                response_id=response_id,
                feedback_store=feedback_store,
            )

    render_feedback_summary(feedback_store)

with evaluation_tab:
    render_evaluation(retriever, top_k)
