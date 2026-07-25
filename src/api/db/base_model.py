"""
Declarative Base and reusable ORM mixins.

All ORM models must inherit from:
  1. Base          — SQLAlchemy DeclarativeBase (2.0 style)
  2. UUIDMixin     — UUID primary key
  3. TimestampMixin — created_at / updated_at with server-side defaults

Using UUID PKs (not integer):
  - Prevents IDOR (Insecure Direct Object Reference) attacks
  - Safe to expose in URLs and API responses
  - Works correctly with distributed / horizontally scaled writes
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    SQLAlchemy 2.0 DeclarativeBase.

    All ORM model classes must subclass this.
    Alembic's env.py imports Base.metadata for auto-generate.
    """


class UUIDMixin:
    """
    UUID primary key mixin.

    Uses PostgreSQL native UUID type (not CHAR/VARCHAR).
    Default generated server-side via gen_random_uuid() for safety,
    with Python-side fallback via uuid.uuid4().
    """
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        nullable=False,
    )


class TimestampMixin:
    """
    Automatic created_at / updated_at timestamp columns.

    created_at: set once on INSERT via server-side NOW()
    updated_at: updated on every UPDATE via server-side NOW()
    also updated Python-side via onupdate for ORM-level updates
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AuditMixin(UUIDMixin, TimestampMixin):
    """
    Convenience mixin combining UUID PK + timestamps.
    Subclass this for most domain tables.
    """
