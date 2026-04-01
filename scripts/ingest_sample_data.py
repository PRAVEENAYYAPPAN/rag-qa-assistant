"""
Script: ingest_sample_data.py
Loads the sample knowledge-base dataset into ChromaDB and FAISS.
Run from the project root:
    python scripts/ingest_sample_data.py
"""

import sys
import json
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ingestion_service import IngestionService
from app.core.logging import setup_logging

setup_logging()

if __name__ == "__main__":
    svc = IngestionService(chunk_size=512, chunk_overlap=64)
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"

    print(f"Ingesting from: {data_dir}")
    result = svc.ingest(source=str(data_dir))
    print(json.dumps(result, indent=2))
