from __future__ import annotations

import structlog
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from atlas_rag.embedding.base import BaseEmbedder

log = structlog.get_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        log.debug("openai_embed", count=len(texts), model=self._model)
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]
