from __future__ import annotations

from typing import List

from src.ingestion.semantic_aliases import alias_query_variants
from src.models import RetrievalResult
from src.retrieval.embeddings import LocalEmbedder
from src.retrieval.query_normalizer import semantic_query_variants
from src.retrieval.vector_store import ChromaVectorStore


class Retriever:
    def __init__(self, embedder: LocalEmbedder, vector_store: ChromaVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        variants = semantic_query_variants(query)
        trusted_alias_start = len(variants)
        if len(variants) == 1:
            for variant in alias_query_variants(query):
                if variant.casefold() not in {item.casefold() for item in variants}:
                    variants.append(variant)
        embeddings = self.embedder.encode(variants)
        best_by_chunk = {}
        original_results = self.vector_store.query(embeddings[0], top_k=top_k)
        for result in original_results:
            best_by_chunk[result.chunk_id] = result

        original_source = original_results[0].source if original_results else None
        for index, embedding in enumerate(embeddings[1:], start=1):
            variant_results = self.vector_store.query(embedding, top_k=top_k)
            requires_source_agreement = index < trusted_alias_start
            if not variant_results or (
                requires_source_agreement
                and variant_results[0].source != original_source
            ):
                continue
            for result in variant_results:
                existing = best_by_chunk.get(result.chunk_id)
                if existing is None or result.distance < existing.distance:
                    best_by_chunk[result.chunk_id] = result

        return sorted(best_by_chunk.values(), key=lambda result: result.distance)[:top_k]
