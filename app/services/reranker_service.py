"""
Re-ranking service using a cross-encoder model.

Cross-encoders jointly encode query + passage and give a more accurate
relevance score than bi-encoder similarity alone.

Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _load_reranker():
    """Load and cache the cross-encoder model once per process."""
    from sentence_transformers.cross_encoder import CrossEncoder   # lazy import
    log.info("Loading re-ranker: {}", settings.RERANKER_MODEL)
    model = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
    log.info("Re-ranker loaded.")
    return model


class RerankerService:
    """Cross-encoder re-ranker that re-scores retrieved candidates."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Re-score *candidates* against *query* using a cross-encoder.

        Args:
            query:      The user's question.
            candidates: List of dicts with at least a ``content`` key.
            top_k:      Return only top-k results; returns all if None.

        Returns:
            Candidates sorted by cross-encoder score (descending),
            with a ``rerank_score`` key added.
        """
        if not candidates:
            return candidates

        _top_k = top_k or settings.TOP_K_RERANK

        model = _load_reranker()
        pairs = [(query, c["content"]) for c in candidates]

        t0 = time.perf_counter()
        scores = model.predict(pairs, show_progress_bar=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        result = ranked[:_top_k]

        log.debug(
            "Re-ranked {} → {} candidates in {:.1f} ms",
            len(candidates),
            len(result),
            elapsed_ms,
        )
        return result


# Module-level singleton
_reranker: RerankerService | None = None


def get_reranker() -> RerankerService:
    global _reranker
    if _reranker is None:
        _reranker = RerankerService()
    return _reranker
