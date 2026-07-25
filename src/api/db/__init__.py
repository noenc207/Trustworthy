"""
Database package public interface.

Exports the engine, session factory, declarative base, mixins,
and session dependency from a single import point so the rest of
the application never needs to know the internal file layout.

Usage:
    from src.api.db import Base, AuditMixin, get_async_session, engine
"""
from src.api.db.base import AsyncSessionLocal, dispose_engine, engine
from src.api.db.base_model import AuditMixin, Base, TimestampMixin, UUIDMixin
from src.api.db.session import get_async_session

__all__ = [
    "AsyncSessionLocal",
    "AuditMixin",
    "Base",
    "TimestampMixin",
    "UUIDMixin",
    "dispose_engine",
    "engine",
    "get_async_session",
]
