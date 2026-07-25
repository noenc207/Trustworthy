"""
Alembic environment for async SQLAlchemy (asyncpg).

This file replaces the standard synchronous env.py that Alembic generates.
Running migrations against an async engine requires a special pattern:
  - Create a synchronous connection from the async engine via
    engine.sync_engine (SQLAlchemy 2.0).
  - All migration operations run inside run_migrations_online() using
    the sync connection wrapper.

The database URL is read from AppSettings (which reads from .env),
so credentials are NEVER committed to alembic.ini.

Usage:
    alembic revision --autogenerate -m "description"
    alembic upgrade head
    alembic downgrade -1
"""
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# ---------------------------------------------------------------------------
# Import application Base so Alembic can discover all ORM models
# for autogenerate. Each model module must be imported here (or imported
# transitively via src.api.db.models) before autogenerate is called.
# ---------------------------------------------------------------------------
from src.api.db.base_model import Base  # noqa: F401 — must be imported for metadata

# Import all model modules here so their tables register on Base.metadata.
from src.api.db.models.user import User          # Milestone 2
from src.api.db.models.upload import Upload      # Milestone 2
from src.api.db.models.prediction import Prediction  # Milestone 2

from src.core.config import get_settings

# ---------------------------------------------------------------------------
# Alembic Config object — access to alembic.ini values
# ---------------------------------------------------------------------------
config = context.config

# Configure Python logging from alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata for autogenerate — Alembic compares this against the live DB
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Read DB URL from AppSettings (never hardcode in alembic.ini)
# ---------------------------------------------------------------------------
settings = get_settings()
_db_url = str(settings.db.url)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    Useful for generating migration SQL to review before applying.
    """
    context.configure(
        url=_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations using a synchronous connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # Include schemas: if you use non-public schemas, list them here
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations using an async engine.

    SQLAlchemy 2.0 requires using engine.sync_engine to obtain a
    synchronous connection for Alembic's migration context.
    """
    connectable = create_async_engine(
        _db_url,
        poolclass=pool.NullPool,  # NullPool: don't reuse connections in migrations
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for 'online' migration mode (default)."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Dispatch based on context mode
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
