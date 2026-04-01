"""
Test configuration / fixtures shared across test modules.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def override_settings(tmp_path_factory):
    """Override ChromaDB and FAISS paths to use temp dirs during tests."""
    from app.core.config import get_settings
    tmp = tmp_path_factory.mktemp("test_chroma")
    settings = get_settings()
    settings.CHROMA_PERSIST_DIR = str(tmp / "chroma")
    settings.FAISS_INDEX_PATH = str(tmp / "faiss.index")
    settings.FAISS_METADATA_PATH = str(tmp / "faiss_meta.json")
    yield
