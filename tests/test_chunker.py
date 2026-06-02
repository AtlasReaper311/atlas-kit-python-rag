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

def test_recursive_splits_large_text():
    sentence = "This is a sentence with some content. "
    text = sentence * 50
    cfg = ChunkConfig(size=200, overlap=20, strategy="recursive")
    chunks = chunk_text(text, cfg)
    assert len(chunks) > 1
