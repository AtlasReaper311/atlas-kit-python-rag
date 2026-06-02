"""
atlas_rag.pipeline
==================
Top-level RAG orchestrator. Composes ingestion, embedding, retrieval, and generation.

Usage:
    from atlas_rag.pipeline import RAGPipeline
    from atlas_rag.config import settings

    pipeline = RAGPipeline.from_settings(settings)
    pipeline.ingest_file("docs/spec.pdf")
    result = pipeline.query("What are the chunking strategies?")
    print(result.answer)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import structlog

from atlas_rag.config import EmbeddingProvider, LLMProvider, Settings
from atlas_rag.embedding.base import BaseEmbedder
from atlas_rag.generation.base import BaseGenerator, GenerationResult
from atlas_rag.ingestion.chunker import ChunkConfig, chunk_text
from atlas_rag.ingestion.loader import load_file
from atlas_rag.store.base import BaseStore, SearchResult

log = structlog.get_logger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseStore,
        generator: BaseGenerator,
        chunk_cfg: ChunkConfig | None = None,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        self._chunk_cfg = chunk_cfg or ChunkConfig()
        self._top_k = top_k

    @classmethod
    def from_settings(cls, cfg: Settings) -> RAGPipeline:
        """Build a fully-wired pipeline from Settings."""
        embedder = _build_embedder(cfg)
        store = _build_store(cfg)
        generator = _build_generator(cfg)
        chunk_cfg = ChunkConfig(size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        return cls(embedder, store, generator, chunk_cfg, top_k=cfg.retrieval_top_k)

    def ingest_file(self, path: str | Path, metadata: dict[str, Any] | None = None) -> int:
        p = Path(path)
        text = load_file(p)
        chunks = chunk_text(text, self._chunk_cfg)

        if not chunks:
            log.warning("ingest_no_chunks", path=str(p))
            return 0

        embeddings = self._embedder.embed(chunks)
        ids = [_chunk_id(str(p), i, chunk) for i, chunk in enumerate(chunks)]
        meta = [{**(metadata or {}), "source": str(p), "chunk_index": i} for i in range(len(chunks))]

        self._store.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=meta)
        log.info("ingested", path=str(p), chunks=len(chunks))
        return len(chunks)

    def ingest_text(self, text: str, source_id: str, metadata: dict[str, Any] | None = None) -> int:
        chunks = chunk_text(text, self._chunk_cfg)
        if not chunks:
            return 0
        embeddings = self._embedder.embed(chunks)
        ids = [_chunk_id(source_id, i, chunk) for i, chunk in enumerate(chunks)]
        meta = [{**(metadata or {}), "source": source_id, "chunk_index": i} for i in range(len(chunks))]
        self._store.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=meta)
        log.info("ingested_text", source=source_id, chunks=len(chunks))
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        q_embedding = self._embedder.embed_one(query)
        return self._store.query(q_embedding, top_k=top_k or self._top_k)

    def query(self, question: str) -> GenerationResult:
        results = self.retrieve(question)
        context = [r.text for r in results]
        log.info("query", question=question[:80], context_chunks=len(context))
        return self._generator.generate(question, context)


def _chunk_id(source: str, index: int, text: str) -> str:
    fingerprint = hashlib.sha256(f"{source}:{index}:{text[:64]}".encode()).hexdigest()[:16]
    return f"{fingerprint}-{index}"


def _build_embedder(cfg: Settings) -> BaseEmbedder:
    if cfg.embedding_provider == EmbeddingProvider.OPENAI:
        from atlas_rag.embedding.openai import OpenAIEmbedder
        return OpenAIEmbedder(api_key=cfg.openai_api_key, model=cfg.embedding_model)  # type: ignore[return-value]
    from atlas_rag.embedding.local import LocalEmbedder
    return LocalEmbedder(model_name=cfg.embedding_model)  # type: ignore[return-value]


def _build_store(cfg: Settings) -> BaseStore:
    from atlas_rag.store.chroma import ChromaStore
    return ChromaStore(host=cfg.chroma_host, port=cfg.chroma_port, collection_name=cfg.chroma_collection)


def _build_generator(cfg: Settings) -> BaseGenerator:
    if cfg.llm_provider == LLMProvider.OPENAI:
        from atlas_rag.generation.openai import OpenAIGenerator
        return OpenAIGenerator(api_key=cfg.openai_api_key, model=cfg.openai_model)  # type: ignore[return-value]
    from atlas_rag.generation.ollama import OllamaGenerator
    return OllamaGenerator(base_url=cfg.ollama_base_url, model=cfg.ollama_model)  # type: ignore[return-value]
