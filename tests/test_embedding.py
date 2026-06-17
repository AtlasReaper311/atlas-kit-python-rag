"""
Tests for atlas_rag.embedding.base.BaseEmbedder.

embed() is abstract and has no behaviour of its own until a real provider
implements it, so this only exercises embed_one's delegation logic via a
trivial concrete subclass. No network, no model download.
"""
from __future__ import annotations

from atlas_rag.embedding.base import BaseEmbedder


class _IdentityLengthEmbedder(BaseEmbedder):
    """Returns [len(text)] per input. Just enough behaviour to prove
    embed_one delegates correctly to embed()."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


def test_embed_one_unwraps_the_first_result() -> None:
    embedder = _IdentityLengthEmbedder()

    result = embedder.embed_one("hello")

    assert result == [5.0]


def test_embed_one_calls_embed_with_a_single_element_batch() -> None:
    calls: list[list[str]] = []

    class _RecordingEmbedder(BaseEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return [[0.0] for _ in texts]

    _RecordingEmbedder().embed_one("only-one")

    assert calls == [["only-one"]]
