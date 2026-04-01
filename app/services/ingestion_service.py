"""
Ingestion service – orchestrates the full pipeline:
  load → chunk → embed → upsert to ChromaDB + FAISS
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store

log = get_logger(__name__)
settings = get_settings()


class IngestionService:
    """Pipeline controller: discover → load → chunk → embed → store."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunker = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._emb = get_embedding_service()
        self._store = get_vector_store()

    # ── Main entry-point ──────────────────────────────────────────────────────

    def ingest(
        self,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a file, directory, or plain text string.

        Returns a summary dict with keys:
          records_processed, chunks_created, collection, status
        """
        path = Path(source)
        documents: list[dict[str, Any]] = []

        if path.is_dir():
            documents = self._load_directory(path)
        elif path.is_file():
            documents = self._load_file(path)
        else:
            # Treat source as raw text
            documents = [{"content": source, "source": "inline_text"}]

        if not documents:
            return {
                "status": "warning",
                "records_processed": 0,
                "chunks_created": 0,
                "collection": settings.CHROMA_COLLECTION_NAME,
                "message": "No documents found or loaded.",
            }

        # Apply extra metadata
        if extra_metadata:
            for doc in documents:
                doc.update(extra_metadata)

        # Limit to MAX_RECORDS
        if len(documents) > settings.MAX_RECORDS:
            log.warning(
                "Capping {} documents to MAX_RECORDS={}",
                len(documents),
                settings.MAX_RECORDS,
            )
            documents = documents[: settings.MAX_RECORDS]

        records_processed = len(documents)

        # Chunk
        chunks = self._chunker.chunk_documents(documents)
        if not chunks:
            return {
                "status": "warning",
                "records_processed": records_processed,
                "chunks_created": 0,
                "collection": settings.CHROMA_COLLECTION_NAME,
                "message": "Chunking produced no output.",
            }

        texts = [c.content for c in chunks]
        metadatas = [c.to_dict() for c in chunks]

        # --- ChromaDB ---
        self._store.add_documents(texts=texts, metadatas=metadatas)

        # --- FAISS (in-memory + persist) ---
        embeddings = self._emb.encode(texts, show_progress=len(texts) > 200)
        faiss_meta = [
            {"content": c.content, "source": c.source, **c.metadata}
            for c in chunks
        ]
        self._emb.build_faiss_index(embeddings, faiss_meta)
        self._emb.save_faiss_index()

        log.info(
            "Ingestion complete: {} records → {} chunks → ChromaDB + FAISS",
            records_processed,
            len(chunks),
        )
        return {
            "status": "success",
            "records_processed": records_processed,
            "chunks_created": len(chunks),
            "collection": settings.CHROMA_COLLECTION_NAME,
            "message": (
                f"Successfully ingested {records_processed} records "
                f"into {len(chunks)} chunks."
            ),
        }

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_directory(self, path: Path) -> list[dict[str, Any]]:
        docs = []
        for ext in settings.SUPPORTED_EXTENSIONS:
            for fpath in path.rglob(f"*{ext}"):
                docs.extend(self._load_file(fpath))
        log.info("Loaded {} documents from directory '{}'", len(docs), path)
        return docs

    def _load_file(self, path: Path) -> list[dict[str, Any]]:
        ext = path.suffix.lower()
        loaders = {
            ".txt": self._load_txt,
            ".md": self._load_txt,
            ".json": self._load_json,
            ".csv": self._load_csv,
        }
        loader = loaders.get(ext)
        if loader is None:
            log.warning("No loader for extension '{}' — skipping {}", ext, path.name)
            return []
        return loader(path)

    def _load_txt(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [{"content": text, "source": str(path.name)}]

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            docs = []
            for item in raw:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("text") or str(item)
                    source = item.get("source", path.name)
                    meta = {k: v for k, v in item.items() if k not in ("content", "text")}
                    docs.append({"content": content, "source": source, **meta})
                else:
                    docs.append({"content": str(item), "source": path.name})
            return docs
        elif isinstance(raw, dict):
            content = raw.get("content") or raw.get("text") or json.dumps(raw)
            return [{"content": content, "source": path.name}]
        return [{"content": str(raw), "source": path.name}]

    def _load_csv(self, path: Path) -> list[dict[str, Any]]:
        docs = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_field = next(
                    (k for k in row if k.lower() in ("content", "text", "body", "description")),
                    None,
                )
                if text_field:
                    content = row.pop(text_field)
                else:
                    content = " | ".join(str(v) for v in row.values())
                docs.append({"content": content, "source": path.name, **dict(row)})
        return docs
