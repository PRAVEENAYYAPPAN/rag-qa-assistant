"""
FastAPI application factory.

Configures:
  - CORS middleware
  - Structured logging
  - Startup / shutdown lifecycle
  - OpenAPI documentation customisation
  - All API routers
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import router

settings = get_settings()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle hooks."""
    # ── Startup ──────────────────────────────────────────────────────────────
    setup_logging()
    log.info("Starting {} v{}", settings.APP_NAME, settings.APP_VERSION)

    # Pre-warm services so first request is fast
    from app.services.embedding_service import get_embedding_service
    from app.services.vector_store import get_vector_store

    emb = get_embedding_service()
    store = get_vector_store()

    # Try to load existing FAISS index
    emb.load_faiss_index()

    log.info(
        "Services ready — ChromaDB docs={}, FAISS={}",
        store.count(),
        "loaded" if emb.faiss_loaded else "empty",
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("Shutting down...")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # ── Root ─────────────────────────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
