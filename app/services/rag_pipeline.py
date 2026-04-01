"""
RAG pipeline – orchestrates end-to-end query answering:
  1. ChromaDB semantic search (bi-encoder retrieval)
  2. Optional cross-encoder re-ranking
  3. LLM answer generation

Returns a fully populated QueryResponse.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import QueryResponse, SourceDocument
from app.services.vector_store import get_vector_store
from app.services.reranker_service import get_reranker
from app.services.llm_service import get_llm_service

log = get_logger(__name__)
settings = get_settings()


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(self) -> None:
        self._store = get_vector_store()
        self._reranker = get_reranker()
        self._llm = get_llm_service()

    def run(
        self,
        question: str,
        top_k: int | None = None,
        rerank: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """
        Execute the full RAG pipeline for a question.

        Args:
            question: Natural-language query.
            top_k:    Number of final context passages; defaults to config.
            rerank:   Whether to apply cross-encoder re-ranking.
            filters:  Optional ChromaDB metadata where filters.

        Returns:
            Fully-populated QueryResponse with answer, sources, and timings.
        """
        _top_k = top_k or settings.TOP_K_RERANK
        t_total_start = time.perf_counter()

        # ── 1. Retrieval ──────────────────────────────────────────────────────
        t_ret_start = time.perf_counter()
        candidates = self._store.search(
            query=question,
            top_k=settings.TOP_K_RETRIEVE,
            where=filters or None,
        )
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        if not candidates:
            log.warning("No documents retrieved for query: '{}'", question[:60])
            return QueryResponse(
                question=question,
                answer=(
                    "I couldn't find relevant information in the knowledge base "
                    "to answer your question."
                ),
                sources=[],
                retrieval_time_ms=round(retrieval_ms, 2),
                generation_time_ms=0.0,
                total_time_ms=round((time.perf_counter() - t_total_start) * 1000, 2),
                model=self._llm.model_name,
                chunks_retrieved=0,
            )

        # ── 2. Re-ranking ─────────────────────────────────────────────────────
        if rerank and settings.RERANKER_ENABLED and len(candidates) > 1:
            passages = self._reranker.rerank(question, candidates, top_k=_top_k)
        else:
            passages = candidates[:_top_k]

        # ── 3. LLM generation ─────────────────────────────────────────────────
        answer, generation_ms = self._llm.answer(question, passages)

        total_ms = (time.perf_counter() - t_total_start) * 1000

        # ── 4. Build response ─────────────────────────────────────────────────
        sources = [
            SourceDocument(
                content=p["content"],
                source=p.get("source", "unknown"),
                score=p.get("rerank_score", p.get("score", 0.0)),
                metadata={
                    k: v
                    for k, v in p.get("metadata", {}).items()
                    if k not in ("content",)
                },
            )
            for p in passages
        ]

        log.info(
            "RAG query complete — retrieved={}, reranked={}, "
            "ret={:.1f}ms gen={:.1f}ms total={:.1f}ms",
            len(candidates), len(passages),
            retrieval_ms, generation_ms, total_ms,
        )

        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_time_ms=round(retrieval_ms, 2),
            generation_time_ms=round(generation_ms, 2),
            total_time_ms=round(total_ms, 2),
            model=self._llm.model_name,
            chunks_retrieved=len(candidates),
        )


# Module-level singleton
_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
