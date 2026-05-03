from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Abstract base class for all embedding providers.

    Implementations: embeddings/bge.py, embeddings/openai.py
    Selected at runtime via EMBEDDING_PROVIDER in .env.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of L2-normalised embedding vectors, one per input text.
            All vectors must have length == settings.embedding_dim.

        Note:
            To embed a single document, use embed_texts([text])[0].
            Use embed_query() for query strings — some providers (e.g. BGE)
            apply a different prompt prefix for queries vs. documents.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Kept separate from embed_texts because some providers (e.g. BGE)
        use a different prompt prefix for queries vs. documents.

        Returns:
            Single L2-normalised embedding vector.
        """
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """
        Return the model version string stored in the DB per chunk.
        Must match EMBEDDING_MODEL_VERSION in .env.
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the output embedding dimension."""
        ...
