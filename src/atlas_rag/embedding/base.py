"""
atlas_rag.embedding.base
========================
Abstract base class for embedding providers.
Swap OpenAI for local (sentence-transformers) by changing one line in config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Embed text into dense vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, same length as `texts`.
        """
        ...

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for single-string embedding."""
        return self.embed([text])[0]
