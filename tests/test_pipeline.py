"""
Tests for atlas_rag.pipeline.RAGPipeline.

Embedder, store, and generator are all hand-rolled fakes, not the real
sentence-transformers / Chroma / Ollama implementations, so this suite
needs no GPU, no running services, and no network. RAGPipeline is built
directly here rather than via from_settings(), which is what lets it skip
the LLM_PROVIDER / CHROMA_HOST environment wiring entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from atlas_rag.pipeline import RAGPipeline


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(t))] for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@dataclass
class FakeSearchResult:
    text: str
    score: float = 1.0


class FakeStore:
    """In-memory stand-in for BaseStore. Returns stored documents on
    query, most-recently-upserted first."""

    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self._documents: list[str] = []

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.upserts.append(
            {"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas}
        )
        self._documents.extend(documents)

    def query(self, query_embedding: list[float], top_k: int) -> list[FakeSearchResult]:
        recent = list(reversed(self._documents))[:top_k]
        return [FakeSearchResult(text=doc) for doc in recent]


@dataclass
class FakeGenerationResult:
    answer: str
    sources: list[str] = field(default_factory=list)


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def generate(self, question: str, context: list[str]) -> FakeGenerationResult:
        self.calls.append((question, context))
        return FakeGenerationResult(answer=f"answer to: {question}", sources=context)


@pytest.fixture
def pipeline() -> tuple[RAGPipeline, FakeEmbedder, FakeStore, FakeGenerator]:
    embedder = FakeEmbedder()
    store = FakeStore()
    generator = FakeGenerator()
    pipe = RAGPipeline(embedder=embedder, store=store, generator=generator, top_k=2)
    return pipe, embedder, store, generator


def test_ingest_text_with_short_string_creates_one_chunk(
    pipeline: tuple[RAGPipeline, FakeEmbedder, FakeStore, FakeGenerator],
) -> None:
    pipe, embedder, store, _ = pipeline

    count = pipe.ingest_text("The quick brown fox jumps over the lazy dog.", source_id="doc-1")

    assert count == 1
    assert len(store.upserts) == 1
    assert len(store.upserts[0]["ids"]) == 1
    assert len(embedder.calls) == 1


def test_ingest_text_empty_string_stores_nothing(
    pipeline: tuple[RAGPipeline, FakeEmbedder, FakeStore, FakeGenerator],
) -> None:
    pipe, embedder, store, _ = pipeline

    count = pipe.ingest_text("", source_id="doc-empty")

    assert count == 0
    assert store.upserts == []
    assert embedder.calls == []


def test_chunk_ids_are_deterministic_for_the_same_input(
    pipeline: tuple[RAGPipeline, FakeEmbedder, FakeStore, FakeGenerator],
) -> None:
    pipe, _, store, _ = pipeline
    pipe.ingest_text("The quick brown fox.", source_id="doc-1")
    first_ids = store.upserts[0]["ids"]

    store2 = FakeStore()
    pipe2 = RAGPipeline(embedder=FakeEmbedder(), store=store2, generator=FakeGenerator())
    pipe2.ingest_text("The quick brown fox.", source_id="doc-1")
    second_ids = store2.upserts[0]["ids"]

    # Same source, same text, same id: re-ingesting an unchanged file is a
    # safe no-op upstream instead of producing a duplicate chunk.
    assert first_ids == second_ids


def test_retrieve_returns_up_to_top_k(
    pipeline: tuple[RAGPipeline, FakeEmbedder, FakeStore, FakeGenerator],
) -> None:
    pipe, embedder, _, _ = pipeline
    pipe.ingest_text("First document about cats.", source_id="doc-1")
    pipe.ingest_text("Second document about dogs.", source_id="doc-2")

    results = pipe.retrieve("tell me about pets")

    assert len(results) == 2  # top_k=2 in the fixture, two docs exist
    assert all(r.text for r in results)
    assert ["tell me about pets"] in embedder.calls


def test_retrieve_respects_an_explicit_top_k_override(
    pipeline: tuple[RAGPipeline, FakeEmbedder, FakeStore, FakeGenerator],
) -> None:
    pipe, _, _, _ = pipeline
    pipe.ingest_text("First document about cats.", source_id="doc-1")
    pipe.ingest_text("Second document about dogs.", source_id="doc-2")

    results = pipe.retrieve("question", top_k=1)

    assert len(results) == 1


def test_query_passes_retrieved_context_to_the_generator(
    pipeline: tuple[RAGPipeline, FakeEmbedder, FakeStore, FakeGenerator],
) -> None:
    pipe, _, _, generator = pipeline
    pipe.ingest_text("The quick brown fox.", source_id="doc-1")

    result = pipe.query("what does this say?")

    assert result.answer == "answer to: what does this say?"
    question, context = generator.calls[0]
    assert question == "what does this say?"
    assert context == result.sources
