"""
Structured logging configuration.
Uses loguru for rich console output (dev) and JSON-structured logs (prod).
"""

import sys
import logging
from loguru import logger
from app.core.config import get_settings


settings = get_settings()


def setup_logging() -> None:
    """Configure loguru with appropriate sinks."""
    logger.remove()

    fmt_dev = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> – "
        "<level>{message}</level>"
    )

    if settings.DEBUG:
        logger.add(sys.stderr, format=fmt_dev, level="DEBUG", colorize=True)
    else:
        logger.add(
            sys.stderr,
            format="{time} | {level} | {name}:{function}:{line} – {message}",
            level=settings.LOG_LEVEL,
            serialize=True,          # JSON in production
        )

    # Also capture stdlib logging into loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back  # type: ignore[assignment]
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for lib_logger in ("uvicorn", "uvicorn.access", "fastapi"):
        logging.getLogger(lib_logger).handlers = [InterceptHandler()]

    logger.info("Logging initialised — level={}", settings.LOG_LEVEL)


def get_logger(name: str):
    """Return a bound loguru logger for a given module name."""
    return logger.bind(module=name)
