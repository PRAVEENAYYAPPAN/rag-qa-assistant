"""
Text chunking service.

Strategies implemented:
  1. recursive  – LangChain-style recursive character splitting (default)
  2. sentence   – NLTK sentence tokenisation
  3. paragraph  – paragraph boundary splitting

Each chunk is returned as a dict: {content, chunk_index, total_chunks, ...meta}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    content: str
    chunk_index: int
    total_chunks: int
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source": self.source,
            **self.metadata,
        }


class ChunkingService:
    """Configurable text chunker with three built-in strategies."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: str | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy or settings.CHUNK_STRATEGY

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split *text* into chunks using the configured strategy."""
        text = text.strip()
        if not text:
            return []

        if self.strategy == "sentence":
            raw_chunks = self._sentence_split(text)
        elif self.strategy == "paragraph":
            raw_chunks = self._paragraph_split(text)
        else:
            raw_chunks = self._recursive_split(text)

        chunks = []
        for i, raw in enumerate(raw_chunks):
            raw = raw.strip()
            if len(raw) < 20:          # skip trivially short chunks
                continue
            chunks.append(
                Chunk(
                    content=raw,
                    chunk_index=i,
                    total_chunks=len(raw_chunks),
                    source=source,
                    metadata=metadata or {},
                )
            )

        log.debug(
            "Chunked '{}…' → {} chunks (strategy={}, size={}, overlap={})",
            text[:40],
            len(chunks),
            self.strategy,
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
        text_key: str = "content",
        source_key: str = "source",
    ) -> list[Chunk]:
        """Chunk a list of document dicts. Returns flattened list of Chunk objects."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            text = doc.get(text_key, "")
            source = doc.get(source_key, "")
            meta = {k: v for k, v in doc.items() if k not in (text_key, source_key)}
            all_chunks.extend(self.chunk_text(text, source=source, metadata=meta))
        log.info(
            "chunk_documents: {} docs → {} total chunks",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    # ── Strategies ────────────────────────────────────────────────────────────

    def _recursive_split(self, text: str) -> list[str]:
        """Recursive character splitting – paragraphs → sentences → words."""
        separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        return self._split_recursive(text, separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._split_by_char(text)

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            return self._split_by_char(text)

        splits = text.split(sep)
        chunks: list[str] = []
        current = ""

        for split in splits:
            candidate = (current + sep + split).lstrip(sep) if current else split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If split itself is too long, recurse
                if len(split) > self.chunk_size:
                    chunks.extend(self._split_recursive(split, remaining))
                    current = ""
                else:
                    current = split

        if current:
            chunks.append(current)

        # Apply overlap: prepend tail of previous chunk
        if self.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_char(self, text: str) -> list[str]:
        """Hard char split as final fallback."""
        step = self.chunk_size - self.chunk_overlap
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), max(step, 1))
        ]

    def _sentence_split(self, text: str) -> list[str]:
        """Split by sentence boundaries, then merge into size-constrained chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return self._apply_overlap(chunks) if self.chunk_overlap > 0 else chunks

    def _paragraph_split(self, text: str) -> list[str]:
        """Split by double-newlines, merge small paragraphs."""
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paras:
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Oversized paragraph → recursive fallback
                if len(para) > self.chunk_size:
                    chunks.extend(self._recursive_split(para))
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the last `chunk_overlap` characters from the previous chunk."""
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                tail = result[-1][-self.chunk_overlap :]
                result.append((tail + " " + chunk).strip())
        return result
