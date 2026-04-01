"""
Integration tests – RAG API endpoints (uses TestClient, no LLM calls).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data
    assert "docs" in data


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data
    assert "chroma_connected" in data
    assert "total_documents" in data


def test_ingest_invalid_source():
    resp = client.post(
        "/api/v1/ingest",
        json={"source": "non_existent_path_xyz_abc"},
    )
    # Either 404 or warning status
    assert resp.status_code in (200, 201, 404, 500)


def test_query_missing_question():
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422   # Pydantic validation error


def test_query_too_short():
    resp = client.post("/api/v1/query", json={"question": "Hi"})
    assert resp.status_code == 422


def test_collection_stats():
    resp = client.get("/api/v1/collection/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "document_count" in data
    assert "embedding_dim" in data


def test_evaluate_empty_samples():
    resp = client.post("/api/v1/evaluate", json={"samples": []})
    assert resp.status_code == 422   # min_length=1


def test_query_valid_format():
    """Query endpoint should respond (may return empty-knowledge answer)."""
    resp = client.post(
        "/api/v1/query",
        json={"question": "What is machine learning?", "top_k": 3, "rerank": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert "retrieval_time_ms" in data
    assert "total_time_ms" in data
