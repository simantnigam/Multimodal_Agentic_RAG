from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from core.config import get_settings
from core.logging import get_logger
from db.models.base import Base

logger = get_logger(__name__)

# Engine and session factory are built lazily on first use so that importing
# this module in tests or scripts does not immediately require a live DB or
# a fully resolved settings object.
_engine: AsyncEngine | None = None
_session_local: async_sessionmaker | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            echo=settings.app_env == "development",   # SQL logging in dev only
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_local
    if _session_local is None:
        _session_local = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,   # avoid lazy-load errors after commit
        )
    return _session_local


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async generator used as a FastAPI dependency.

    Usage:
        async def my_route(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Create all tables that do not yet exist.
    Called once at app startup in app/main.py.

    Note: does not apply schema.sql migrations — use db/migrations/ for
    production schema changes (indexes, triggers, extensions).
    """
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialised", tables=list(Base.metadata.tables.keys()))
