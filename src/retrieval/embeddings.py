from __future__ import annotations

from typing import Iterable, List

from src.config import EMBEDDING_MODEL, MODEL_CACHE_DIR


class LocalEmbedder:
    """Lazy-loading Sentence Transformers embedding adapter."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        allow_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.allow_download = allow_download
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(MODEL_CACHE_DIR),
                    local_files_only=True,
                )
            except OSError as error:
                if not self.allow_download:
                    raise RuntimeError(
                        "Embedding model is not available locally. Run the index "
                        "command once while connected to the internet."
                    ) from error
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(MODEL_CACHE_DIR),
                )
        return self._model

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        values = list(texts)
        if not values:
            return []
        embeddings = self.model.encode(
            values,
            normalize_embeddings=True,
            show_progress_bar=len(values) > 8,
        )
        return embeddings.tolist()
