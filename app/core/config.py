"""
Central configuration management using Pydantic BaseSettings.
All values can be overridden via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path
from typing import Literal


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "RAG Q&A Assistant"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Production-grade Retrieval-Augmented Generation API "
        "with semantic search, ChromaDB persistence, and FAISS indexing."
    )
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── API Keys ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    COHERE_API_KEY: str = ""          # optional – re-ranking
    HUGGINGFACE_TOKEN: str = ""       # optional – local models

    # ── LLM provider ─────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["openai", "groq", "local"] = "groq"
    LLM_MODEL: str = "llama3-8b-8192"   # groq default; swap to gpt-4o for openai
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    # ── Embedding model ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    EMBEDDING_BATCH_SIZE: int = 64

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    CHUNK_STRATEGY: Literal["recursive", "sentence", "paragraph"] = "recursive"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K_RETRIEVE: int = 10          # candidates from vector store
    TOP_K_RERANK: int = 4             # final passages after re-ranking
    SIMILARITY_THRESHOLD: float = 0.25
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_store")
    CHROMA_COLLECTION_NAME: str = "rag_knowledge_base"

    # ── FAISS ─────────────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = str(BASE_DIR / "data" / "processed" / "faiss.index")
    FAISS_METADATA_PATH: str = str(BASE_DIR / "data" / "processed" / "faiss_meta.json")

    # ── Data ingestion ────────────────────────────────────────────────────────
    RAW_DATA_DIR: str = str(BASE_DIR / "data" / "raw")
    PROCESSED_DATA_DIR: str = str(BASE_DIR / "data" / "processed")
    SUPPORTED_EXTENSIONS: list[str] = [".txt", ".pdf", ".json", ".csv", ".md"]
    MAX_RECORDS: int = 5000

    # ── Evaluation ───────────────────────────────────────────────────────────
    EVAL_RECALL_K: int = 5
    EVAL_PRECISION_K: int = 5

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
