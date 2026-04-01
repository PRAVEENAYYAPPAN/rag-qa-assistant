"""
ChromaDB vector store service.

Handles:
  - Collection creation / retrieval (persistent client)
  - Upsert (add or update) documents with embeddings
  - Semantic similarity search with optional metadata filters
  - Collection statistics
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedding_service import get_embedding_service

log = get_logger(__name__)
settings = get_settings()


class VectorStoreService:
    """Production ChromaDB service with full CRUD + semantic search."""

    def __init__(self) -> None:
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._emb_svc = get_embedding_service()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Initialise persistent ChromaDB client and get / create collection."""
        log.info("Connecting ChromaDB — persist_dir={}", settings.CHROMA_PERSIST_DIR)
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaDB ready — collection='{}', documents={}",
            settings.CHROMA_COLLECTION_NAME,
            self._collection.count(),
        )

    def _ensure_connected(self) -> None:
        if self._collection is None:
            self.connect()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """Embed and upsert documents into the collection.

        Args:
            texts:     List of raw text chunks.
            metadatas: Parallel list of metadata dicts.
            ids:       Optional stable IDs; UUIDs generated if omitted.

        Returns:
            Number of documents added.
        """
        self._ensure_connected()
        if not texts:
            return 0

        _ids = ids or [str(uuid.uuid4()) for _ in texts]
        _metas = metadatas or [{}] * len(texts)

        log.info("Embedding {} chunks for ChromaDB upsert…", len(texts))
        embeddings = self._emb_svc.encode(texts, show_progress=len(texts) > 200).tolist()

        # Upsert in batches of 512 (Chroma ceiling)
        batch_size = 512
        for i in range(0, len(texts), batch_size):
            self._collection.upsert(  # type: ignore[union-attr]
                ids=_ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                metadatas=_metas[i : i + batch_size],
            )

        log.info("ChromaDB upsert complete — total={}", self._collection.count())  # type: ignore
        return len(texts)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search with optional metadata filtering.

        Returns list of dicts: {content, score, source, metadata}.
        """
        self._ensure_connected()
        query_emb = self._emb_svc.encode_query(query).tolist()
        _threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results": min(top_k, max(1, self._collection.count())),  # type: ignore
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)  # type: ignore

        output = []
        docs = results["documents"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0]  # type: ignore

        for doc, dist, meta in zip(docs, dists, metas):
            # ChromaDB returns cosine distance; convert to similarity
            score = 1.0 - float(dist)
            if score < _threshold:
                continue
            output.append(
                {
                    "content": doc,
                    "score": round(score, 4),
                    "source": meta.get("source", "unknown"),
                    "metadata": meta,
                }
            )

        log.debug(
            "ChromaDB search: query='{}…' → {} results (threshold={})",
            query[:40],
            len(output),
            _threshold,
        )
        return output

    # ── Stats & Admin ─────────────────────────────────────────────────────────

    def count(self) -> int:
        self._ensure_connected()
        return self._collection.count()  # type: ignore

    def delete(self, ids: list[str]) -> None:
        self._ensure_connected()
        self._collection.delete(ids=ids)  # type: ignore
        log.info("Deleted {} documents from ChromaDB", len(ids))

    def reset_collection(self) -> None:
        """Drop and recreate the collection (destructive!)."""
        self._ensure_connected()
        self._client.delete_collection(settings.CHROMA_COLLECTION_NAME)  # type: ignore
        self._collection = self._client.get_or_create_collection(  # type: ignore
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.warning("Collection '{}' reset.", settings.CHROMA_COLLECTION_NAME)

    @property
    def connected(self) -> bool:
        return self._client is not None


# Module-level singleton
_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
        _vector_store.connect()
    return _vector_store
