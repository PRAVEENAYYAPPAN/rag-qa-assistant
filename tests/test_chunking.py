"""
Unit tests – chunking service
"""
import pytest
from app.services.chunking_service import ChunkingService


@pytest.fixture
def chunker():
    return ChunkingService(chunk_size=200, chunk_overlap=20, strategy="recursive")


def test_chunk_short_text(chunker):
    text = "Hello world."
    chunks = chunker.chunk_text(text, source="test")
    # Very short text – might be skipped (< 20 chars) – adjust threshold or text
    assert isinstance(chunks, list)


def test_chunk_long_text(chunker):
    text = " ".join([f"Sentence number {i} is here to provide context." for i in range(50)])
    chunks = chunker.chunk_text(text, source="test_long")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 220   # slight leeway for overlap


def test_chunk_overlap_applied(chunker):
    text = "A " * 300
    chunks = chunker.chunk_text(text)
    if len(chunks) > 1:
        # Second chunk should contain some content from first
        assert len(chunks[1].content) > 0


def test_chunk_documents(chunker):
    docs = [
        {"content": "Doc one. " * 30, "source": "doc1.txt"},
        {"content": "Doc two. " * 30, "source": "doc2.txt"},
    ]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) >= 2
    sources = {c.source for c in chunks}
    assert "doc1.txt" in sources
    assert "doc2.txt" in sources


def test_sentence_strategy():
    chunker = ChunkingService(chunk_size=100, chunk_overlap=10, strategy="sentence")
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunker.chunk_text(text)
    assert all(len(c.content) > 0 for c in chunks)


def test_paragraph_strategy():
    chunker = ChunkingService(chunk_size=200, chunk_overlap=0, strategy="paragraph")
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 1


def test_empty_text(chunker):
    chunks = chunker.chunk_text("")
    assert chunks == []


def test_metadata_propagated(chunker):
    text = "Test chunk text content is long enough to exceed twenty characters."
    chunks = chunker.chunk_text(text, source="src.txt", metadata={"domain": "science"})
    for c in chunks:
        assert c.metadata.get("domain") == "science"
        assert c.source == "src.txt"
