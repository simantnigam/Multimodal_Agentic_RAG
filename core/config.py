# Requires: uv add pydantic-settings
from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import (
    MODEL_DIM_MAP,
    EMBEDDING_PROVIDER_BGE,
    EMBEDDING_PROVIDER_OPENAI,
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # App
    # -------------------------------------------------------------------------
    app_env:        str = "development"
    app_host:       str = "0.0.0.0"
    app_port:       int = 8000
    app_workers:    int = 4
    log_level:      str = "INFO"

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url:       str = "postgresql+asyncpg://admin:admin@localhost:5432/rag_db"
    db_pool_size:       int = 10
    db_max_overflow:    int = 10

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    redis_url:          str = "redis://localhost:6379/0"
    cache_ttl_seconds:  int = 3600

    # -------------------------------------------------------------------------
    # Celery
    # -------------------------------------------------------------------------
    celery_broker_url:      str = "redis://localhost:6379/1"
    celery_result_backend:  str = "redis://localhost:6379/2"
    celery_queue_ingestion: str = QUEUE_INGESTION
    celery_queue_embedding: str = QUEUE_EMBEDDING
    celery_queue_storage:   str = QUEUE_STORAGE
    celery_max_retries:     int = 3
    celery_retry_delay:     int = 60

    # -------------------------------------------------------------------------
    # LLM
    # -------------------------------------------------------------------------
    anthropic_api_key:  str   = ""
    llm_model:          str   = "claude-sonnet-4-6"
    llm_max_tokens:     int   = 1000
    llm_temperature:    float = 0.0
    openai_api_key:     str   = ""

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    embedding_provider:         str = EMBEDDING_PROVIDER_BGE
    embedding_model:            str = "BAAI/bge-large-en-v1.5"
    embedding_model_version:    str = "bge-large-en-v1.5"
    embedding_dim:              int = 1024
    embedding_batch_size:       int = 64

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------
    chunking_similarity_threshold:  float = CHUNKING_SIMILARITY_THRESHOLD
    chunk_min_tokens:               int   = CHUNK_MIN_TOKENS
    chunk_max_tokens:               int   = CHUNK_MAX_TOKENS

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    top_k_pre_rerank:   int   = DEFAULT_TOP_K_PRE_RERANK
    top_k_post_rerank:  int   = DEFAULT_TOP_K_POST_RERANK
    dense_weight:       float = DEFAULT_DENSE_WEIGHT
    sparse_weight:      float = DEFAULT_SPARSE_WEIGHT
    ivfflat_probes:     int   = IVFFLAT_PROBES

    # -------------------------------------------------------------------------
    # Agents & Orchestration
    # -------------------------------------------------------------------------
    validator_threshold:    float = VALIDATOR_THRESHOLD
    max_retries:            int   = MAX_RETRIES
    context_max_tokens:     int   = CONTEXT_MAX_TOKENS
    query_expansion_count:  int   = QUERY_EXPANSION_COUNT

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------
    langsmith_api_key:  str  = ""
    langsmith_project:  str  = "multimodal-rag"
    langsmith_tracing:  bool = False    # opt-in: requires LANGSMITH_API_KEY to be set
    prometheus_port:    int  = 9090

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    eval_golden_dataset_path:   str = "data/golden_dataset.json"
    eval_precision_k:           int = 5
    eval_llm_judge_model:       str = "claude-sonnet-4-6"

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def check_embedding_consistency(self) -> "Settings":
        expected_dim = MODEL_DIM_MAP.get(self.embedding_model)
        if expected_dim is not None and self.embedding_dim != expected_dim:
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} does not match "
                f"model '{self.embedding_model}' (expected {expected_dim}). "
                f"Update EMBEDDING_DIM in your .env."
            )
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
        return self

    @model_validator(mode="after")
    def check_embedding_version_sync(self) -> "Settings":
        # embedding_model_version is stored per chunk in the DB.
        # It must be a suffix-match of embedding_model to avoid silent drift.
        # e.g. model=BAAI/bge-large-en-v1.5 → version must contain bge-large-en-v1.5
        if self.embedding_model_version not in self.embedding_model:
            raise ValueError(
                f"EMBEDDING_MODEL_VERSION='{self.embedding_model_version}' does not match "
                f"EMBEDDING_MODEL='{self.embedding_model}'. "
                f"Update EMBEDDING_MODEL_VERSION in your .env."
            )
        return self

    @model_validator(mode="after")
    def check_hybrid_weights(self) -> "Settings":
        if abs(self.dense_weight + self.sparse_weight - 1.0) > 0.001:
            raise ValueError(
                f"DENSE_WEIGHT + SPARSE_WEIGHT must equal 1.0 "
                f"(got {self.dense_weight + self.sparse_weight:.3f}). Check your .env."
            )
        return self

    @model_validator(mode="after")
    def check_chunk_token_bounds(self) -> "Settings":
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            raise ValueError(
                f"CHUNK_MIN_TOKENS ({self.chunk_min_tokens}) must be less than "
                f"CHUNK_MAX_TOKENS ({self.chunk_max_tokens}). Check your .env."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
