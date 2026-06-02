"""
atlas_rag.api.app
=================
FastAPI application factory. Call create_app() to get a configured instance.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from atlas_rag.api.routes import router
from atlas_rag.config import settings
from atlas_rag.pipeline import RAGPipeline

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build the pipeline once at startup; share it via app.state."""
    log.info("atlas_rag_starting", provider_llm=settings.llm_provider, provider_embed=settings.embedding_provider)
    app.state.pipeline = RAGPipeline.from_settings(settings)
    log.info("atlas_rag_ready")
    yield
    log.info("atlas_rag_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Atlas RAG",
        version="0.1.0",
        description="Production RAG pipeline — atlas-systems.uk",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
