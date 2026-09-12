"""
core/database.py
================
Async SQLAlchemy engine and session factory.

Architecture decision:
- We use async SQLAlchemy (asyncpg driver) for all FastAPI route handlers
  so request handlers never block the event loop on DB I/O.
- A sync engine is also available for Alembic migrations and scripts that
  don't run inside an async context.
- get_db() is an async generator injected via FastAPI Depends().
  It guarantees the session is closed even if the route raises an exception.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------
# pool_pre_ping=True: issues a SELECT 1 before handing a connection from the
# pool, so stale connections (e.g. after Docker restart) are detected and
# replaced instead of causing cryptic errors in route handlers.
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=(settings.APP_ENV == "development"),  # SQL logging in dev only
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Avoids lazy-load errors after commit in async code
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Base class for all ORM models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """
    All SQLAlchemy ORM models must inherit from this Base.
    Placing it here (not in models/) avoids circular imports.
    """
    pass


# ---------------------------------------------------------------------------
# Dependency for FastAPI routes
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.

    Usage in a route:
        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------
async def check_database_connection() -> bool:
    """
    Attempt a lightweight query to confirm the database is reachable.
    Used by the /api/system/health endpoint.
    """
    from sqlalchemy import text

    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Context manager for use in scripts / background tasks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Use this in scripts and background tasks that are not inside a FastAPI
    request context and therefore cannot use Depends().

    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
