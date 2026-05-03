"""
Celery tasks for async document ingestion.

Workers are started with:
    celery -A ingestion.tasks worker --loglevel=info -Q ingestion,embedding,storage

The pipeline is async (SQLAlchemy asyncpg) but Celery workers are sync.
asyncio.run() bridges the gap — acceptable for ingestion which is not latency-sensitive.

Pool compatibility:
    asyncio.run() requires the default prefork pool.
    Do NOT use --pool=gevent or --pool=eventlet — they monkey-patch the stdlib
    and will cause asyncio.run() to raise "This event loop is already running".
"""
from __future__ import annotations

import asyncio
from typing import Any

from celery import Celery

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Celery app — configured from settings
# ---------------------------------------------------------------------------

def _make_celery() -> Celery:
    from dotenv import load_dotenv
    load_dotenv()          # ensure .env is loaded before get_settings() caches values
    settings = get_settings()
    app = Celery(
        "rag_worker",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,          # re-queue on worker crash
        worker_prefetch_multiplier=1, # one task at a time per worker
    )
    return app


celery_app = _make_celery()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_pipeline(doc_path: str, metadata: dict[str, Any]) -> str:
    """Async wrapper that owns the session and optional Redis client."""
    from db.session import get_session_factory
    from ingestion.pipeline import ingest_document

    # Optional Redis for embedding cache
    redis_client = None
    try:
        import redis as redis_lib
        settings = get_settings()
        redis_client = redis_lib.from_url(settings.redis_url, decode_responses=False)
        redis_client.ping()
    except Exception:
        logger.warning("redis_unavailable_for_cache", note="Continuing without embedding cache")
        redis_client = None

    async with get_session_factory()() as session:
        try:
            doc_id = await ingest_document(
                path=doc_path,
                metadata=metadata,
                session=session,
                redis_client=redis_client,
            )
            await session.commit()
            return doc_id
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="ingestion.tasks.ingest_document_task",
    queue="ingestion",
    max_retries=None,           # retry count read from settings at runtime
    default_retry_delay=None,   # retry delay read from settings at runtime
)
def ingest_document_task(
    self,
    doc_path:  str,
    metadata:  dict[str, Any] | None = None,
) -> str:
    """
    Celery task: ingest a single document end-to-end.

    Args:
        doc_path: Absolute path to the document file.
        metadata: Optional extra metadata attached to the document record.

    Returns:
        doc_id (UUID string) of the stored document.

    Retry policy:
        - Retries up to CELERY_MAX_RETRIES times (default: 3).
        - Exponential backoff: delay doubles on each retry (base: CELERY_RETRY_DELAY).
        - After all retries exhausted the task fails and is available in the result backend.
    """
    settings = get_settings()
    metadata = metadata or {}

    try:
        logger.info(
            "ingest_task_start",
            doc_path=doc_path,
            attempt=self.request.retries + 1,
        )
        doc_id = asyncio.run(_run_pipeline(doc_path, metadata))
        logger.info("ingest_task_complete", doc_id=doc_id, doc_path=doc_path)
        return doc_id

    except Exception as exc:
        attempt   = self.request.retries + 1
        max_retries = settings.celery_max_retries
        base_delay  = settings.celery_retry_delay
        delay       = base_delay * (2 ** self.request.retries)   # exponential backoff

        logger.warning(
            "ingest_task_failed",
            doc_path=doc_path,
            attempt=attempt,
            max_retries=max_retries,
            retry_in=delay,
            error=str(exc),
        )

        if self.request.retries < max_retries:
            raise self.retry(exc=exc, countdown=delay)

        # Final failure — log and let Celery mark the task as FAILURE
        logger.error(
            "ingest_task_dead",
            doc_path=doc_path,
            attempts=attempt,
            error=str(exc),
        )
        raise
