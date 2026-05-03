import hashlib
import json

import redis

from core.config import get_settings
from core.interfaces.embedder import Embedder
from core.logging import get_logger

logger = get_logger(__name__)


def _make_key(text: str, model_version: str) -> str:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"emb:{model_version}:{content_hash}"


def _get(r: redis.Redis, key: str) -> list[float] | None:
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        # Cache read failure is non-fatal — fall through to embed
        logger.warning("embedding_cache_read_failed", key=key, error=str(exc))
        return None


def _set(r: redis.Redis, key: str, embedding: list[float], ttl: int) -> None:
    try:
        r.setex(key, ttl, json.dumps(embedding))
    except Exception as exc:
        # Cache write failure is non-fatal — embedding already computed
        logger.warning("embedding_cache_write_failed", key=key, error=str(exc))


def embed_with_cache(
    texts: list[str],
    embedder: Embedder,
    r: redis.Redis,
) -> list[list[float]]:
    """
    Embed texts using the cache when available.

    Cache key includes the model version so that swapping models
    never serves stale embeddings from the previous model.

    Cache failures (read or write) are silently absorbed — the pipeline
    continues by calling the embedder directly.
    """
    if not texts:
        return []

    settings = get_settings()
    ttl = settings.cache_ttl_seconds
    model_version = embedder.model_version

    results: list[list[float] | None] = [None] * len(texts)
    cache_misses: list[int] = []

    # Check cache for each text
    for i, text in enumerate(texts):
        key = _make_key(text, model_version)
        cached = _get(r, key)
        if cached is not None:
            results[i] = cached
        else:
            cache_misses.append(i)

    # Batch embed all cache misses in one call
    if cache_misses:
        miss_texts = [texts[i] for i in cache_misses]
        embeddings = embedder.embed_texts(miss_texts)

        for idx, embedding in zip(cache_misses, embeddings):
            results[idx] = embedding
            key = _make_key(texts[idx], model_version)
            _set(r, key, embedding, ttl)

    logger.debug(
        "embedding_cache_stats",
        total=len(texts),
        hits=len(texts) - len(cache_misses),
        misses=len(cache_misses),
    )

    return results  # type: ignore[return-value]
