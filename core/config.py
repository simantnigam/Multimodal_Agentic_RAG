import os
from functools import lru_cache
from dotenv import load_dotenv

from core.constants import (
    MODEL_DIM_MAP,
    EMBEDDING_PROVIDER_BGE,
    EMBEDDING_PROVIDER_OPENAI,
    VISION_PROVIDER_ANTHROPIC,
    CHUNK_MIN_TOKENS,
    CHUNK_MAX_TOKENS,
    CHUNKING_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K_PRE_RERANK,
    DEFAULT_TOP_K_POST_RERANK,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_SPARSE_WEIGHT,
    VALIDATOR_THRESHOLD,
    MAX_RETRIES,
    CONTEXT_MAX_TOKENS,
    QUERY_EXPANSION_COUNT,
    IVFFLAT_PROBES,
    QUEUE_INGESTION,
    QUEUE_EMBEDDING,
    QUEUE_STORAGE,
)


class Settings:
    """
    Loads configuration from environment variables (via .env).
    load_dotenv() strips inline comments before populating os.environ,
    so values like `APP_PORT=8000  # comment` resolve correctly to `8000`.
    """

    def __init__(self) -> None:
        load_dotenv(override=False)  # won't overwrite vars already set in the shell

        # ---------------------------------------------------------------------
        # App
        # ---------------------------------------------------------------------
        self.app_env        = os.getenv("APP_ENV",      "development")
        self.app_host       = os.getenv("APP_HOST",     "0.0.0.0")
        self.app_port       = int(os.getenv("APP_PORT",     "8000"))
        self.app_workers    = int(os.getenv("APP_WORKERS",  "4"))
        self.log_level      = os.getenv("LOG_LEVEL",    "INFO").upper()

        # ---------------------------------------------------------------------
        # Database
        # ---------------------------------------------------------------------
        self.database_url    = os.getenv("DATABASE_URL",    "postgresql+asyncpg://admin:admin@localhost:5432/rag_db")
        self.db_pool_size    = int(os.getenv("DB_POOL_SIZE",    "10"))
        self.db_max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

        # ---------------------------------------------------------------------
        # Redis
        # ---------------------------------------------------------------------
        self.redis_url         = os.getenv("REDIS_URL",          "redis://localhost:6379/0")
        self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

        # ---------------------------------------------------------------------
        # Celery
        # ---------------------------------------------------------------------
        self.celery_broker_url      = os.getenv("CELERY_BROKER_URL",     "redis://localhost:6379/1")
        self.celery_result_backend  = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
        self.celery_queue_ingestion = os.getenv("CELERY_QUEUE_INGESTION", QUEUE_INGESTION)
        self.celery_queue_embedding = os.getenv("CELERY_QUEUE_EMBEDDING", QUEUE_EMBEDDING)
        self.celery_queue_storage   = os.getenv("CELERY_QUEUE_STORAGE",   QUEUE_STORAGE)
        self.celery_max_retries     = int(os.getenv("CELERY_MAX_RETRIES", "3"))
        self.celery_retry_delay     = int(os.getenv("CELERY_RETRY_DELAY", "60"))

        # ---------------------------------------------------------------------
        # LLM
        # ---------------------------------------------------------------------
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.llm_model         = os.getenv("LLM_MODEL",         "claude-sonnet-4-6")
        self.llm_max_tokens    = int(os.getenv("LLM_MAX_TOKENS",   "1000"))
        self.llm_temperature   = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.openai_api_key    = os.getenv("OPENAI_API_KEY", "")

        # ---------------------------------------------------------------------
        # Vision (image captioning)
        # ---------------------------------------------------------------------
        # Provider auto-detected if not set: anthropic if key present, else openai, else none
        self.vision_provider    = os.getenv("VISION_PROVIDER",    self._detect_vision_provider())
        self.vision_model       = os.getenv("VISION_MODEL",       "claude-sonnet-4-6")
        self.vision_max_tokens  = int(os.getenv("VISION_MAX_TOKENS", "300"))

        # ---------------------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------------------
        self.embedding_provider      = os.getenv("EMBEDDING_PROVIDER",      EMBEDDING_PROVIDER_BGE)
        self.embedding_model         = os.getenv("EMBEDDING_MODEL",          "BAAI/bge-large-en-v1.5")
        self.embedding_model_version = os.getenv("EMBEDDING_MODEL_VERSION",  "bge-large-en-v1.5")
        self.embedding_dim           = int(os.getenv("EMBEDDING_DIM",        "1024"))
        self.embedding_batch_size    = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

        # ---------------------------------------------------------------------
        # Chunking
        # ---------------------------------------------------------------------
        self.chunking_similarity_threshold = float(os.getenv("CHUNKING_SIMILARITY_THRESHOLD", str(CHUNKING_SIMILARITY_THRESHOLD)))
        self.chunk_min_tokens              = int(os.getenv("CHUNK_MIN_TOKENS", str(CHUNK_MIN_TOKENS)))
        self.chunk_max_tokens              = int(os.getenv("CHUNK_MAX_TOKENS", str(CHUNK_MAX_TOKENS)))

        # ---------------------------------------------------------------------
        # Retrieval
        # ---------------------------------------------------------------------
        self.top_k_pre_rerank  = int(os.getenv("TOP_K_PRE_RERANK",  str(DEFAULT_TOP_K_PRE_RERANK)))
        self.top_k_post_rerank = int(os.getenv("TOP_K_POST_RERANK", str(DEFAULT_TOP_K_POST_RERANK)))
        self.dense_weight      = float(os.getenv("DENSE_WEIGHT",    str(DEFAULT_DENSE_WEIGHT)))
        self.sparse_weight     = float(os.getenv("SPARSE_WEIGHT",   str(DEFAULT_SPARSE_WEIGHT)))
        self.ivfflat_probes    = int(os.getenv("IVFFLAT_PROBES",    str(IVFFLAT_PROBES)))

        # ---------------------------------------------------------------------
        # Agents & Orchestration
        # ---------------------------------------------------------------------
        self.validator_threshold   = float(os.getenv("VALIDATOR_THRESHOLD",  str(VALIDATOR_THRESHOLD)))
        self.max_retries           = int(os.getenv("MAX_RETRIES",            str(MAX_RETRIES)))
        self.context_max_tokens    = int(os.getenv("CONTEXT_MAX_TOKENS",     str(CONTEXT_MAX_TOKENS)))
        self.query_expansion_count = int(os.getenv("QUERY_EXPANSION_COUNT",  str(QUERY_EXPANSION_COUNT)))

        # ---------------------------------------------------------------------
        # Observability
        # ---------------------------------------------------------------------
        self.langsmith_api_key  = os.getenv("LANGSMITH_API_KEY",  "")
        self.langsmith_project  = os.getenv("LANGSMITH_PROJECT",  "multimodal-rag")
        self.langsmith_tracing  = os.getenv("LANGSMITH_TRACING",  "false").strip().lower() == "true"
        self.prometheus_port    = int(os.getenv("PROMETHEUS_PORT", "9090"))

        # ---------------------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------------------
        self.eval_golden_dataset_path = os.getenv("EVAL_GOLDEN_DATASET_PATH", "data/golden_dataset.json")
        self.eval_precision_k         = int(os.getenv("EVAL_PRECISION_K",      "5"))
        self.eval_llm_judge_model     = os.getenv("EVAL_LLM_JUDGE_MODEL",      "claude-sonnet-4-6")

        self._validate()

    def _detect_vision_provider(self) -> str:
        """Auto-detect vision provider from available API keys if VISION_PROVIDER not set."""
        if os.getenv("ANTHROPIC_API_KEY", ""):
            return VISION_PROVIDER_ANTHROPIC
        if os.getenv("OPENAI_API_KEY", ""):
            return "openai"
        return "none"

    def _validate(self) -> None:
        # Embedding model <-> dim consistency
        expected_dim = MODEL_DIM_MAP.get(self.embedding_model)
        if expected_dim is not None and self.embedding_dim != expected_dim:
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} does not match "
                f"model '{self.embedding_model}' (expected {expected_dim}). "
                f"Update EMBEDDING_DIM in your .env."
            )

        # Provider <-> model family consistency
        if self.embedding_provider == EMBEDDING_PROVIDER_BGE and "BAAI" not in self.embedding_model:
            raise ValueError(
                f"EMBEDDING_PROVIDER=bge but EMBEDDING_MODEL='{self.embedding_model}' "
                f"is not a BGE model. Check your .env."
            )
        if self.embedding_provider == EMBEDDING_PROVIDER_OPENAI and "BAAI" in self.embedding_model:
            raise ValueError(
                f"EMBEDDING_PROVIDER=openai but EMBEDDING_MODEL='{self.embedding_model}' "
                f"is a BGE model. Check your .env."
            )

        # Model name <-> version sync
        if self.embedding_model_version not in self.embedding_model:
            raise ValueError(
                f"EMBEDDING_MODEL_VERSION='{self.embedding_model_version}' does not match "
                f"EMBEDDING_MODEL='{self.embedding_model}'. "
                f"Update EMBEDDING_MODEL_VERSION in your .env."
            )

        # Hybrid weights must sum to 1.0
        if abs(self.dense_weight + self.sparse_weight - 1.0) > 0.001:
            raise ValueError(
                f"DENSE_WEIGHT + SPARSE_WEIGHT must equal 1.0 "
                f"(got {self.dense_weight + self.sparse_weight:.3f}). Check your .env."
            )

        # Chunk token bounds
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            raise ValueError(
                f"CHUNK_MIN_TOKENS ({self.chunk_min_tokens}) must be less than "
                f"CHUNK_MAX_TOKENS ({self.chunk_max_tokens}). Check your .env."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
