import time
import numpy as np
import openai

from core.config import get_settings
from core.logging import get_logger
from core.interfaces.embedder import Embedder

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0    # seconds; doubles on each retry


class OpenAIEmbedder(Embedder):
    """
    Embedder backed by the OpenAI Embeddings API.
    Requires OPENAI_API_KEY and EMBEDDING_MODEL set to an OpenAI model
    (e.g. text-embedding-3-large).

    OpenAI embeddings do not require a query instruction prefix —
    embed_query() and embed_texts() use the same encoding path.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._model_version = settings.embedding_model_version
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for batch in self._batched(texts):
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        # OpenAI uses the same model for queries and documents — no prefix needed.
        return self._call_api([query])[0]

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                )
                embeddings = [item.embedding for item in response.data]
                return [self._normalise(e) for e in embeddings]

            except openai.AuthenticationError as exc:
                # Bad API key — retrying will never help
                raise RuntimeError(
                    "OpenAI authentication failed. Check OPENAI_API_KEY in your .env."
                ) from exc

            except openai.RateLimitError as exc:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "openai_rate_limit",
                    attempt=attempt + 1,
                    retry_in=delay,
                )
                time.sleep(delay)
                last_exc = exc

            except openai.APIError as exc:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "openai_api_error",
                    attempt=attempt + 1,
                    error=str(exc),
                    retry_in=delay,
                )
                time.sleep(delay)
                last_exc = exc

        raise RuntimeError(
            f"OpenAI embedding failed after {_MAX_RETRIES} attempts"
        ) from last_exc

    def _normalise(self, vec: list[float]) -> list[float]:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def _batched(self, texts: list[str]) -> list[list[str]]:
        return [
            texts[i: i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
