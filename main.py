"""
Uvicorn entrypoint – run with:  python main.py
or production:                  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import uvicorn
from app.main import app
from app.core.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
