from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.models import Chunk, RetrievalResult


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl=(
                    "src.retrieval.telemetry.NoopProductTelemetry"
                ),
            ),
        )
        self.collection_name = collection_name

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection")

        collection = self._get_or_create_collection()
        existing_ids = collection.get(include=[])["ids"]
        new_ids = [chunk.id for chunk in chunks]
        stale_ids = sorted(set(existing_ids) - set(new_ids))
        if stale_ids:
            collection.delete(ids=stale_ids)
        collection.upsert(
            ids=new_ids,
            embeddings=[list(embedding) for embedding in embeddings],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "character_count": chunk.character_count,
                }
                for chunk in chunks
            ],
        )

    def count(self) -> int:
        return self._get_or_create_collection().count()

    def query(self, query_embedding: Sequence[float], top_k: int) -> List[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        collection = self._get_or_create_collection()
        count = collection.count()
        if count == 0:
            raise RuntimeError("The index is empty. Run the index command first.")

        response = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                source=str(metadata["source"]),
                chunk_index=int(metadata["chunk_index"]),
                distance=float(distance),
            )
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]
