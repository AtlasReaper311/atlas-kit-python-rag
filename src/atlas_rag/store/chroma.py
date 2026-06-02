"""
atlas_rag.store.chroma
======================
ChromaDB vector store. Implements the abstract BaseStore interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import chromadb
import structlog

from atlas_rag.config import settings
from atlas_rag.store.base import BaseStore, SearchResult

log = structlog.get_logger(__name__)


@dataclass
class ChromaStore(BaseStore):
    host: str = settings.chroma_host
    port: int = settings.chroma_port
    collection_name: str = settings.chroma_collection

    def __post_init__(self) -> None:
        self._client = chromadb.HttpClient(host=self.host, port=self.port)
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("chroma_connected", host=self.host, port=self.port, collection=self.collection_name)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._col.upsert(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        log.debug("chroma_upsert", count=len(ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = settings.retrieval_top_k,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)

        docs = cast(list[list[str]], results["documents"])
        dists = cast(list[list[float]], results["distances"])
        metas = cast(list[list[dict[str, Any]]], results["metadatas"])

        out: list[SearchResult] = []
        for doc, dist, meta in zip(docs[0], dists[0], metas[0], strict=False):
            out.append(SearchResult(text=doc, score=1.0 - dist, metadata=meta))
        return out

    def count(self) -> int:
        return self._col.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
        log.warning("chroma_collection_deleted", collection=self.collection_name)
