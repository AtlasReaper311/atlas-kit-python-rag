from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict[str, Any]] | None = None) -> None: ...

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int, where: dict[str, Any] | None = None) -> list[SearchResult]: ...

    @abstractmethod
    def count(self) -> int: ...
