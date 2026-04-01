#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Render.com start script for RAG Q&A Assistant
# 1. Auto-ingests sample data if the knowledge base is empty
# 2. Starts the FastAPI server on $PORT (injected by Render)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "🚀 Starting RAG Q&A Assistant..."

# Auto-seed the knowledge base on cold start
echo "📚 Seeding knowledge base..."
python scripts/ingest_sample_data.py || echo "⚠️  Ingestion skipped (may already be populated)"

# Start FastAPI with uvicorn on Render's dynamic $PORT
echo "🌐 Starting server on port ${PORT:-8000}..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level info
