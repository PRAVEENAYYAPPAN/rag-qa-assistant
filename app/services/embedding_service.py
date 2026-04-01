"""
Embedding service: wraps sentence-transformers and caches the model singleton.

Supports:
  - Batch encode with progress bar
  - Normalised L2 embeddings (cosine similarity via dot-product)
  - FAISS index creation and persistence
"""

from __future__ import annotations

import json
import time
import numpy as np
import faiss
from pathlib import Path
from typing import Optional
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model once per process."""
    log.info("Loading embedding model: {}", settings.EMBEDDING_MODEL)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    log.info("Embedding model loaded — dim={}", model.get_sentence_embedding_dimension())
    return model


class EmbeddingService:
    """Thin wrapper around SentenceTransformer with FAISS integration."""

    def __init__(self) -> None:
        self.model = _load_model()
        self._dim = self.model.get_sentence_embedding_dimension()
        self._index: Optional[faiss.IndexFlatIP] = None
        self._meta: list[dict] = []

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts → float32 array of shape (N, dim).
        Vectors are L2-normalised so that dot-product == cosine similarity.
        """
        bs = batch_size or settings.EMBEDDING_BATCH_SIZE
        t0 = time.perf_counter()
        vectors = self.model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        log.debug(
            "Encoded {} texts in {:.1f} ms (batch_size={})",
            len(texts), elapsed, bs,
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → 1-D float32 array."""
        return self.encode([query])[0]

    # ── FAISS index ───────────────────────────────────────────────────────────

    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """Build an in-memory FAISS inner-product index from embeddings."""
        index = faiss.IndexFlatIP(self._dim)
        index.add(embeddings)
        self._index = index
        self._meta = metadata
        log.info("Built FAISS index — {} vectors, dim={}", index.ntotal, self._dim)

    def save_faiss_index(self) -> None:
        """Persist FAISS index + metadata to disk."""
        if self._index is None:
            raise RuntimeError("No FAISS index in memory. Call build_faiss_index first.")
        idx_path = Path(settings.FAISS_INDEX_PATH)
        meta_path = Path(settings.FAISS_METADATA_PATH)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(idx_path))
        meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2))
        log.info("FAISS index saved → {}", idx_path)

    def load_faiss_index(self) -> bool:
        """Load FAISS index from disk. Returns True on success."""
        idx_path = Path(settings.FAISS_INDEX_PATH)
        meta_path = Path(settings.FAISS_METADATA_PATH)
        if not idx_path.exists():
            log.warning("FAISS index not found at {}", idx_path)
            return False
        self._index = faiss.read_index(str(idx_path))
        self._meta = json.loads(meta_path.read_text())
        log.info("FAISS index loaded — {} vectors", self._index.ntotal)
        return True

    def faiss_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Search FAISS index with a query string.

        Returns list of dicts with keys: content, score, metadata.
        """
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("FAISS index empty or not loaded.")
        vec = self.encode_query(query).reshape(1, -1)
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._meta[idx]
            results.append(
                {
                    "content": meta.get("content", ""),
                    "score": float(score),
                    "metadata": {k: v for k, v in meta.items() if k != "content"},
                }
            )
        return results

    @property
    def faiss_loaded(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def dim(self) -> int:
        return self._dim


# Module-level singleton
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
