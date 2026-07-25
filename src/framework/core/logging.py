"""
Structured JSON Logging Infrastructure.
"""
from __future__ import annotations

import json
import logging


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Correlate traces if present
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "execution_id"):
            log_obj["execution_id"] = record.execution_id
        if hasattr(record, "trace_id"):
            log_obj["trace_id"] = record.trace_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)

class LoggingInfrastructure:
    """Configures structured JSON logging with trace correlation."""

    @staticmethod
    def setup(level: str = "INFO") -> None:
        logger = logging.getLogger()
        logger.setLevel(level)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)
