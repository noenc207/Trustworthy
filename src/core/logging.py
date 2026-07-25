"""
Structured logging configuration using Loguru.
Supports both development (human-readable) and production (JSON) modes.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.core.config import AppSettings


def setup_logging(settings: AppSettings) -> None:
    """
    Configure Loguru logger.

    Development: colorized, human-readable output.
    Production: structured JSON for log aggregation (e.g. Loki, CloudWatch).
    """
    logger.remove()  # Remove default handler

    if settings.app_env == "production":
        # Structured JSON logging for production
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} | {message}",
            level=settings.log_level,
            serialize=True,  # JSON output
            enqueue=True,     # Thread-safe
            backtrace=False,
            diagnose=False,
        )
        # Rotate file logs
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            compression="gz",
            level="WARNING",
            serialize=True,
        )
    else:
        # Colorized human-readable for development
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.log_level,
            backtrace=True,
            diagnose=True,
        )
