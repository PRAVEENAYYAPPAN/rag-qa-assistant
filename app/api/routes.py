"""
FastAPI routers – /ingest, /query, /evaluate, /health, /collection
"""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException, status, BackgroundTasks

from app.models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    EvalRequest, EvalResponse,
    HealthResponse, CollectionStats, DeleteRequest,
)
from app.services.ingestion_service import IngestionService
from app.services.rag_pipeline import get_rag_pipeline
from app.services.evaluation_service import EvaluationService
from app.services.vector_store import get_vector_store
from app.services.embedding_service import get_embedding_service
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

router = APIRouter()


# ── /ingest ───────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documents into the knowledge base",
    tags=["Ingestion"],
)
async def ingest_documents(body: IngestRequest) -> IngestResponse:
    """
    Load, chunk, embed, and store documents from a file path, directory,
    or raw text string.

    - Supports: **.txt, .md, .json, .csv**
    - Chunks with configurable size / overlap
    - Persists to both **ChromaDB** and **FAISS**
    """
    try:
        svc = IngestionService(
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
        result = svc.ingest(source=body.source, extra_metadata=body.metadata)
        return IngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("Ingestion error: {}", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# ── /query ────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the knowledge base",
    tags=["Query"],
)
async def query(body: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
    1. Semantic retrieval from ChromaDB (bi-encoder)
    2. Optional cross-encoder re-ranking
    3. LLM answer generation with context injection

    Returns the answer, grounding sources, and latency breakdowns.
    """
    pipeline = get_rag_pipeline()
    try:
        return pipeline.run(
            question=body.question,
            top_k=body.top_k,
            rerank=body.rerank,
            filters=body.filters or None,
        )
    except Exception as e:
        log.exception("Query error: {}", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ── /evaluate ─────────────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvalResponse,
    summary="Evaluate retrieval quality (P@K, R@K, F1, MRR, NDCG)",
    tags=["Evaluation"],
)
async def evaluate(body: EvalRequest) -> EvalResponse:
    """
    Run precision/recall-based evaluation over a labelled sample set.

    Each sample requires a `question` and optionally `relevant_doc_ids`
    (list of expected document IDs). Returns aggregate and per-sample metrics.
    """
    try:
        svc = EvaluationService()
        return svc.evaluate(body)
    except Exception as e:
        log.exception("Evaluation error: {}", e)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


# ── /health ───────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and status",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Returns liveness / readiness status of all subsystems."""
    store = get_vector_store()
    emb = get_embedding_service()
    try:
        doc_count = store.count()
        chroma_ok = True
    except Exception:
        doc_count = -1
        chroma_ok = False

    return HealthResponse(
        status="healthy" if chroma_ok else "degraded",
        version=settings.APP_VERSION,
        chroma_connected=chroma_ok,
        faiss_loaded=emb.faiss_loaded,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
        total_documents=doc_count,
    )


# ── /collection ───────────────────────────────────────────────────────────────

@router.get(
    "/collection/stats",
    response_model=CollectionStats,
    summary="ChromaDB collection statistics",
    tags=["System"],
)
async def collection_stats() -> CollectionStats:
    """Return metadata and document count for the active ChromaDB collection."""
    store = get_vector_store()
    emb = get_embedding_service()
    return CollectionStats(
        name=settings.CHROMA_COLLECTION_NAME,
        document_count=store.count(),
        embedding_dim=emb.dim,
        persist_dir=settings.CHROMA_PERSIST_DIR,
    )


@router.delete(
    "/collection/documents",
    summary="Delete specific documents by ID",
    tags=["System"],
    status_code=status.HTTP_200_OK,
)
async def delete_documents(body: DeleteRequest) -> dict:
    """Permanently delete documents from ChromaDB by their IDs."""
    store = get_vector_store()
    try:
        store.delete(body.ids)
        return {"deleted": len(body.ids), "ids": body.ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/collection/reset",
    summary="Reset (clear) the entire collection",
    tags=["System"],
    status_code=status.HTTP_200_OK,
)
async def reset_collection() -> dict:
    """⚠️ Destructive – drops and recreates the ChromaDB collection."""
    store = get_vector_store()
    store.reset_collection()
    return {"status": "reset", "collection": settings.CHROMA_COLLECTION_NAME}
