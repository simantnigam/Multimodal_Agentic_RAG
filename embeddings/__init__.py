from functools import lru_cache

from core.config import get_settings
from core.constants import EMBEDDING_PROVIDER_BGE, EMBEDDING_PROVIDER_OPENAI
from core.interfaces.embedder import Embedder


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    Return the configured embedder instance.

    Cached after first call — loading the BGE model (~1.3 GB) is expensive
    and must only happen once per process.

    Provider is selected via EMBEDDING_PROVIDER in .env:
        bge    → BGEEmbedder   (local, no API key)
        openai → OpenAIEmbedder (API, requires OPENAI_API_KEY)
    """
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == EMBEDDING_PROVIDER_BGE:
        from embeddings.bge import BGEEmbedder
        return BGEEmbedder()

    if provider == EMBEDDING_PROVIDER_OPENAI:
        from embeddings.openai import OpenAIEmbedder
        return OpenAIEmbedder()

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{provider}'. "
        f"Valid options: '{EMBEDDING_PROVIDER_BGE}', '{EMBEDDING_PROVIDER_OPENAI}'."
    )
