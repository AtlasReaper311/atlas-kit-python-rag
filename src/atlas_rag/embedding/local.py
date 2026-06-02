from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

from atlas_rag.embedding.base import BaseEmbedder

log = structlog.get_logger(__name__)


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        log.info("loading_local_embedder", model=model_name)
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()  # type: ignore[return-value]
