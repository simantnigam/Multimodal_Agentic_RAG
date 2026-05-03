from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.interfaces.embedder import Embedder
from core.logging import get_logger

logger = get_logger(__name__)

# BGE-large prepend this prefix for query strings to align the query
# embedding space with the document embedding space.
# Documents are embedded without any prefix.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder(Embedder):
    """
    Embedder backed by a local BAAI/bge-* model via sentence-transformers.
    No API key required — model runs entirely on-device.
    """

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("bge_model_loading", model=settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model)
        self._model_version = settings.embedding_model_version
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        logger.info("bge_model_ready", model=settings.embedding_model, dim=self._dim)

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode(
            _QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dim(self) -> int:
        return self._dim

