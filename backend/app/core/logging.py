"""
core/logging.py
===============
Structured logging configuration for SensorShield.
Uses Python's standard logging with a consistent format so log lines
are machine-parseable in production (and readable during development).
"""

import logging
import sys
from typing import Optional

from app.core.config import settings


def configure_logging(level: Optional[str] = None) -> None:
    """
    Call once at application startup (in main.py lifespan).
    Sets the root logger and suppresses noisy third-party loggers.
    """
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
        force=True,
    )

    # Quieten noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.APP_ENV == "development" else logging.WARNING
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper — use in every module:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
