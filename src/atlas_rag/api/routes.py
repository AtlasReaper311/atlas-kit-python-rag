"""
atlas_rag.api.routes
====================
REST endpoints:

  POST /ingest/text        Ingest raw text
  POST /query              Query the RAG pipeline
  GET  /retrieve           Retrieve chunks without generating
  GET  /health             Liveness + store stats
"""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from atlas_rag.store.base import SearchResult

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to ingest")
    source_id: str = Field(..., description="Identifier for this document (e.g. filename, URL)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    source_id: str
    chunks_stored: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    model: str
    total_tokens: int


class RetrieveResponse(BaseModel):
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    store_count: int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(body: IngestTextRequest, request: Request) -> IngestResponse:
    pipeline = request.app.state.pipeline
    try:
        n = pipeline.ingest_text(body.text, body.source_id, body.metadata)
    except Exception as e:
        log.error("ingest_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    return IngestResponse(source_id=body.source_id, chunks_stored=n)


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    pipeline = request.app.state.pipeline
    try:
        result = pipeline.query(body.question)
    except Exception as e:
        log.error("query_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    return QueryResponse(answer=result.answer, model=result.model, total_tokens=result.total_tokens)


@router.get("/retrieve", response_model=RetrieveResponse)
async def retrieve(q: str, request: Request, top_k: int = 5) -> RetrieveResponse:
    pipeline = request.app.state.pipeline
    results: list[SearchResult] = pipeline.retrieve(q, top_k=top_k)
    return RetrieveResponse(
        results=[{"text": r.text, "score": round(r.score, 4), "metadata": r.metadata} for r in results]
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    pipeline = request.app.state.pipeline
    count = pipeline._store.count()
    return HealthResponse(status="ok", store_count=count)
