"""
atlas_rag.ingestion.chunker
===========================
Text chunking strategies. Three modes:

- fixed:      Simple character-count windows. Fast, predictable.
- recursive:  Tries paragraph > sentence > word boundaries before falling back.
- semantic:   Placeholder for embedding-based boundary detection.

All chunkers return a list of strings, never empty strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from atlas_rag.config import settings


ChunkStrategy = Literal["fixed", "recursive", "semantic"]

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class ChunkConfig:
    size: int = settings.chunk_size
    overlap: int = settings.chunk_overlap
    strategy: ChunkStrategy = "recursive"


def chunk_text(text: str, cfg: ChunkConfig | None = None) -> list[str]:
    """Entry point. Delegates to the appropriate strategy."""
    if not text.strip():
        return []
    c = cfg or ChunkConfig()
    if c.strategy == "fixed":
        return _fixed(text, c.size, c.overlap)
    return _recursive(text, c.size, c.overlap, _SEPARATORS)


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _recursive(text: str, size: int, overlap: int, separators: list[str]) -> list[str]:
    separator = separators[-1]
    for sep in separators:
        if sep and sep in text:
            separator = sep
            break

    splits = text.split(separator) if separator else list(text)
    splits = [s for s in splits if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for split in splits:
        split_len = len(split)
        if current_len + split_len > size and current:
            merged = separator.join(current).strip()
            if merged:
                chunks.append(merged)
            while current and current_len > overlap:
                removed = current.pop(0)
                current_len -= len(removed) + len(separator)
        current.append(split)
        current_len += split_len + len(separator)

    if current:
        merged = separator.join(current).strip()
        if merged:
            chunks.append(merged)

    final: list[str] = []
    next_seps = separators[separators.index(separator) + 1:] if separator in separators else separators
    for chunk in chunks:
        if len(chunk) > size and next_seps:
            final.extend(_recursive(chunk, size, overlap, next_seps))
        else:
            final.append(chunk)

    return final
