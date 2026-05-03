# =============================================================================
# Chunking
# =============================================================================

CHUNK_MIN_TOKENS = 150
CHUNK_MAX_TOKENS = 600
CHUNKING_SIMILARITY_THRESHOLD = 0.75


# =============================================================================
# Modality
# =============================================================================

MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_TABLE = "table"

MODALITIES = {MODALITY_TEXT, MODALITY_IMAGE, MODALITY_TABLE}


# =============================================================================
# Embedding
# =============================================================================

EMBEDDING_DIM_BGE_LARGE = 1024
EMBEDDING_DIM_BGE_BASE = 768
EMBEDDING_DIM_OAI_SMALL = 1536
EMBEDDING_DIM_OAI_LARGE = 3072

MODEL_DIM_MAP: dict[str, int] = {
    "BAAI/bge-large-en-v1.5":  EMBEDDING_DIM_BGE_LARGE,
    "BAAI/bge-base-en-v1.5":   EMBEDDING_DIM_BGE_BASE,
    "text-embedding-3-small":   EMBEDDING_DIM_OAI_SMALL,
    "text-embedding-3-large":   EMBEDDING_DIM_OAI_LARGE,
}

EMBEDDING_PROVIDER_BGE = "bge"
EMBEDDING_PROVIDER_OPENAI = "openai"


# =============================================================================
# Retrieval
# =============================================================================

DEFAULT_TOP_K_PRE_RERANK = 20
DEFAULT_TOP_K_POST_RERANK = 5
DEFAULT_DENSE_WEIGHT = 0.7
DEFAULT_SPARSE_WEIGHT = 0.3


# =============================================================================
# Validation
# =============================================================================

VALIDATOR_THRESHOLD = 0.75
MAX_RETRIES = 2

DECISION_PASS = "pass"
DECISION_RETRY = "retry"
DECISION_DEGRADE = "degrade"


# =============================================================================
# Relationships
# =============================================================================

RELATION_CHUNK_TO_ASSET = "chunk_to_asset"
RELATION_CHUNK_TO_CHUNK = "chunk_to_chunk"


# =============================================================================
# Document types
# =============================================================================

DOC_TYPE_PDF = "pdf"
DOC_TYPE_HTML = "html"
DOC_TYPE_DOCX = "docx"
DOC_TYPE_IMAGE = "image"

DOC_TYPES = {DOC_TYPE_PDF, DOC_TYPE_HTML, DOC_TYPE_DOCX, DOC_TYPE_IMAGE}


# =============================================================================
# Agents & Orchestration
# =============================================================================

CONTEXT_MAX_TOKENS = 4000
QUERY_EXPANSION_COUNT = 3


# =============================================================================
# Database
# =============================================================================

IVFFLAT_PROBES = 10


# =============================================================================
# Celery queues
# =============================================================================

QUEUE_INGESTION = "ingestion"
QUEUE_EMBEDDING = "embedding"
QUEUE_STORAGE = "storage"
