from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: list[str]) -> GenerationResult: ...
