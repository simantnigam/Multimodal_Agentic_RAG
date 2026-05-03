-- =============================================================================
-- Multimodal Agentic RAG — Production Schema
-- =============================================================================
-- Idempotent: safe to run multiple times (uses IF NOT EXISTS throughout)
-- Requires: pgvector extension (installed automatically below)
-- Embedding model: BAAI/bge-large-en-v1.5 → 1024 dimensions
-- =============================================================================


-- =============================================================================
-- Extensions
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- =============================================================================
-- Core Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS documents (
    doc_id      UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    title       TEXT,
    source      TEXT,
    doc_type    TEXT,                       -- pdf | html | docx | image
    created_at  TIMESTAMP   DEFAULT NOW(),
    metadata    JSONB
);


CREATE TABLE IF NOT EXISTS sections (
    section_id          UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id              UUID    NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_section_id   UUID,               -- NULL for top-level sections
    title               TEXT,
    level               INT,                -- heading depth: 1=H1, 2=H2, ...
    metadata            JSONB
);


CREATE TABLE IF NOT EXISTS chunks (
    chunk_id                UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id                  UUID        NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id              UUID        REFERENCES sections(section_id),
    content                 TEXT        NOT NULL,
    embedding               VECTOR(1024),
    embedding_model_version TEXT        DEFAULT 'bge-large-en-v1.5',
    modality                TEXT        DEFAULT 'text',     -- text | image | table
    token_count             INT,
    metadata                JSONB,
    tsv                     TSVECTOR,                       -- populated by trigger below
    created_at              TIMESTAMP   DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS assets (
    asset_id    UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id      UUID    NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    type        TEXT,                       -- image | table
    content     TEXT,                       -- alt-text, caption, or table markdown
    embedding   VECTOR(1024),
    metadata    JSONB
);


CREATE TABLE IF NOT EXISTS relationships (
    id              SERIAL  PRIMARY KEY,
    source_id       UUID    NOT NULL,       -- chunk_id or asset_id
    target_id       UUID    NOT NULL,       -- chunk_id or asset_id
    relation_type   TEXT    NOT NULL        -- chunk_to_asset | chunk_to_chunk
);


-- =============================================================================
-- Evaluation Log Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id                  SERIAL      PRIMARY KEY,
    trace_id            UUID,
    query               TEXT,
    retrieved_chunk_ids UUID[],
    scores              FLOAT[],
    latency_ms          INT,
    created_at          TIMESTAMP   DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS evaluation_logs (
    id                  SERIAL      PRIMARY KEY,
    trace_id            UUID,
    query               TEXT,
    answer              TEXT,
    groundedness_score  FLOAT,
    completeness_score  FLOAT,
    hallucination_score FLOAT,
    final_score         FLOAT,
    retry_count         INT         DEFAULT 0,
    confidence          FLOAT,
    latency_breakdown   JSONB,      -- {"query_agent": ms, "retrieval": ms, ...}
    feedback            TEXT,
    created_at          TIMESTAMP   DEFAULT NOW()
);


-- =============================================================================
-- Indexes
-- =============================================================================

-- Vector similarity search (ANN) — ivfflat for cosine distance
-- lists=100 suits corpora up to ~100K chunks; tune in Phase 8
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- BM25 full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON chunks USING GIN(tsv);

-- FK + filter lookups
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON chunks (doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_section_id
    ON chunks (section_id);

CREATE INDEX IF NOT EXISTS idx_sections_doc_id
    ON sections (doc_id);

CREATE INDEX IF NOT EXISTS idx_assets_doc_id
    ON assets (doc_id);

-- Relationship lookups in both directions (context assembly needs target_id)
CREATE INDEX IF NOT EXISTS idx_relationships_src
    ON relationships (source_id);

CREATE INDEX IF NOT EXISTS idx_relationships_tgt
    ON relationships (target_id);

-- Evaluation log lookups
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_trace
    ON retrieval_logs (trace_id);

CREATE INDEX IF NOT EXISTS idx_evaluation_logs_trace
    ON evaluation_logs (trace_id);


-- =============================================================================
-- BM25 Trigger — auto-populate tsv on insert/update
-- =============================================================================

CREATE OR REPLACE FUNCTION update_tsv()
RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS tsv_update ON chunks;

CREATE TRIGGER tsv_update
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_tsv();
