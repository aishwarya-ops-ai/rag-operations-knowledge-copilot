from __future__ import annotations

from typing import List

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR
from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import load_txt_documents
from src.ingestion.semantic_aliases import add_retrieval_aliases
from src.models import Chunk


def prepare_chunks() -> List[Chunk]:
    documents = load_txt_documents(DOCUMENTS_DIR)
    return add_retrieval_aliases(
        chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
    )
