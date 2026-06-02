from __future__ import annotations

from atlas_rag.ingestion.chunker import ChunkConfig, chunk_text


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_fixed_chunks_cover_all_content():
    text = "a" * 1000
    cfg = ChunkConfig(size=200, overlap=0, strategy="fixed")
    chunks = chunk_text(text, cfg)
    assert len(chunks) == 5


def test_fixed_overlap_produces_more_chunks():
    text = "a" * 100
    cfg = ChunkConfig(size=40, overlap=10, strategy="fixed")
    chunks = chunk_text(text, cfg)
    assert len(chunks) > 3


def test_recursive_splits_large_text():
    sentence = "This is a sentence with some content. "
    text = sentence * 50
    cfg = ChunkConfig(size=200, overlap=20, strategy="recursive")
    chunks = chunk_text(text, cfg)
    assert len(chunks) > 1


def test_all_chunks_are_strings():
    text = "Some text to chunk up into pieces for testing purposes."
    cfg = ChunkConfig(size=20, overlap=5, strategy="recursive")
    chunks = chunk_text(text, cfg)
    assert all(isinstance(c, str) for c in chunks)


def test_chunks_are_deterministic():
    text = "Deterministic chunking test. Same input should give same output."
    cfg = ChunkConfig(size=512, overlap=0, strategy="fixed")
    assert chunk_text(text, cfg) == chunk_text(text, cfg)
