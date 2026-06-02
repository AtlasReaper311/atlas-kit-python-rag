from __future__ import annotations

import structlog
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from atlas_rag.generation.base import BaseGenerator, GenerationResult

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a precise, factual assistant. Answer the user's question using only the
provided context. If the context does not contain enough information to answer
confidently, say so — do not fabricate details.

Context:
{context}
"""


class OpenAIGenerator(BaseGenerator):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, query: str, context_chunks: list[str]) -> GenerationResult:
        context = "\n\n---\n\n".join(context_chunks)
        log.debug("openai_generate", model=self._model, context_chunks=len(context_chunks))
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        return GenerationResult(
            answer=answer,
            model=self._model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )
