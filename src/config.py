from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "models"
EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation_questions.csv"
PARAPHRASE_EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation_paraphrases.csv"
EVALUATION_RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "results.jsonl"
FEEDBACK_FILE = Path(
    os.getenv(
        "OPS_COPILOT_FEEDBACK_FILE",
        str(PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"),
    )
)

COLLECTION_NAME = "operations_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4

INSUFFICIENT_ANSWER = "I could not find enough information in the knowledge base."
MIN_GROUNDING_SIMILARITY = 0.50
HIGH_CONFIDENCE_SIMILARITY = 0.65
OLLAMA_BASE_URL = os.getenv("OPS_COPILOT_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OPS_COPILOT_OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT_SECONDS = 120
