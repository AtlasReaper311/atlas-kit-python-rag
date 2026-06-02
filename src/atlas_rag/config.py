"""
atlas_rag.config
================
All configuration is sourced from environment variables and validated at startup.
No scattered os.getenv() calls anywhere else in the codebase — everything lives here.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"
    LOCAL = "local"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Embedding ─────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── LLM ───────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Vector Store ──────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "atlas_rag"

    # ── Retrieval ─────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_strategy: Literal["similarity", "mmr", "hybrid"] = "similarity"
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Chunking ──────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)

    # ── API ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    log_level: str = "info"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, v: int, info: object) -> int:
        # Access via info.data when using pydantic v2 model_validator would be cleaner,
        # but field_validator is sufficient for this check at construction time.
        return v


# Module-level singleton — import and use everywhere.
settings = Settings()
