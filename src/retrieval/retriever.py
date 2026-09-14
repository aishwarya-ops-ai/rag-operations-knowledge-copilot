from __future__ import annotations

from typing import List

from src.models import RetrievalResult
from src.retrieval.embeddings import LocalEmbedder
from src.retrieval.vector_store import ChromaVectorStore


class Retriever:
    def __init__(self, embedder: LocalEmbedder, vector_store: ChromaVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
        query_embedding = self.embedder.encode([query])[0]
        return self.vector_store.query(query_embedding, top_k=top_k)

