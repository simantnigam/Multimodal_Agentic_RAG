# Multimodal Agentic RAG System
## Extended LLD + Architecture + Sequences + Eval + Deployment

---

# 1. End-to-End Query Lifecycle (Sequence Diagram)

```
User
 ↓
API Gateway (FastAPI + Rate Limiter + Auth)
 ↓
Orchestrator (LangGraph)
 ↓
Query Agent → Query Rewrite → Query Expansion (synonyms, multi-query)
 ↓
Retrieval Agent
    ├─ Dense Retrieval (pgvector ANN)
    ├─ Sparse Retrieval (BM25 / tsvector)
    └─ Metadata Filter (domain, doc_type, date range)
 ↓
Hybrid Fusion (α·dense_norm + β·bm25_norm)  ← scores normalized before fusion
 ↓
Reranker (Cross Encoder)
 ↓
Context Assembly
    ├─ Expand with same-section chunks
    ├─ Include linked assets (images, tables)
    ├─ Deduplicate overlapping content
    └─ Truncate to token limit (~4000 tokens)
 ↓
Answer Generator (LLM)
 ↓
Validator Agent
    ├─ Groundedness Check
    ├─ Hallucination Detection
    └─ Completeness Check
 ↓
IF fail (max 2 retries) → Feedback Loop (Re-retrieval / Query Expansion)
IF fail after retries   → Return partial answer with confidence flag
 ↓
Final Response
 ↓
Evaluator Agent → Logs Metrics → LangSmith Trace
```

---

# 2. LangGraph Node-Level Mapping

## Graph Nodes

```python
nodes = {
    "query_agent":          query_agent,          # rewrite + expand query
    "retrieval_agent":      retrieval_agent,       # dense + sparse + filter
    "reranker":             reranker,              # cross-encoder scoring
    "context_assembly":     context_assembly,      # expand, dedup, truncate context
    "answer_agent":         answer_agent,          # LLM generation
    "validator":            validator_agent,       # groundedness + hallucination
    "evaluator":            evaluator_agent,       # metrics logging
    "query_expander_agent": query_expander_agent,  # fallback multi-query path
}
```

## Edges

```python
query_agent          → retrieval_agent
retrieval_agent      → reranker
reranker             → context_assembly
context_assembly     → answer_agent
answer_agent         → validator
validator            → evaluator        # PASS: final_score >= 0.75
validator            → query_agent      # FAIL (retry_count < 2): re-retrieve
validator            → evaluator        # FAIL (retry_count >= 2): degrade gracefully
evaluator            → END
```

## State Schema

```python
class RAGState(TypedDict):
    query:             str
    rewritten_query:   str
    expanded_queries:  list[str]
    retrieved_chunks:  list[Chunk]
    reranked_chunks:   list[Chunk]
    context_window:    str          # assembled, deduplicated, truncated context
    answer:            str
    validator_scores:  dict         # {"groundedness": f, "completeness": f, "hallucination": f, "final_score": f}
    validation_pass:   bool
    retry_count:       int          # caps at 2 to prevent infinite loops
    confidence:        float
    trace_id:          str
    metadata:          dict
    latency_breakdown: dict         # {"query_agent": ms, "retrieval": ms, "reranker": ms, "context_assembly": ms, "llm_answer": ms, "validator": ms}
```

---

# 3. Agent Specifications

## 3.1 Query Agent

**Responsibility:** Rewrite the raw user query for retrieval quality.

**Token Budget:** ~500 input + 200 output tokens

**Steps:**
1. Detect query type (factual, analytical, comparative, multimodal)
2. Rewrite for clarity and retrieval intent
3. Pass to Query Expander for multi-query generation

**Error Handling:**
- If LLM call fails → fall back to raw query unchanged
- Log rewrite failure with trace_id

---

## 3.2 Query Expander Agent

**Responsibility:** Generate multiple query variants to improve recall.

**Token Budget:** ~600 input + 400 output tokens

**Strategies:**
- Synonym expansion via WordNet / LLM
- Intent classification (narrow → expand; broad → narrow)
- Generate 3–5 sub-queries for multi-query retrieval

**Error Handling:**
- If expansion fails → use original rewritten query only
- Do not block the pipeline

---

## 3.3 Retrieval Agent

**Responsibility:** Execute hybrid retrieval across dense and sparse indexes.

**Token Budget:** N/A (database queries, not LLM)

**Steps:**
1. Run dense ANN search (pgvector `<=>` operator)
2. Run BM25 sparse search (tsvector `ts_rank`)
3. Apply metadata filters (domain, date, doc_type)
4. Normalize scores independently before fusion (min-max per result set)
5. Fuse: `final_score = 0.7 * dense_norm + 0.3 * bm25_norm`
6. Return Top-K (default K=20 pre-rerank, 5 post-rerank)

**Error Handling:**
- If DB unavailable → raise `RetrieverUnavailableError`, abort pipeline
- If zero results after query expansion → set `answer="No relevant information found"`, `confidence=0.0`, route directly to evaluator to log failure with trace_id
- Timeout: 3 seconds per retrieval call

---

## 3.4 Context Assembly

**Responsibility:** Build the final context window passed to the Answer Agent.

**Steps:**
1. Take Top-K reranked chunks
2. Expand: fetch additional chunks from the same `section_id`
3. Expand: fetch linked assets (images, tables) via `relationships` table
4. Deduplicate by `chunk_id`
5. Order: highest relevance first; maintain document order within same section
6. Truncate to max ~4000 tokens

**Output:** A single assembled `context_window` string stored in `RAGState`.

---

## 3.5 Answer Agent

**Responsibility:** Generate a grounded answer from retrieved context.

**Token Budget:** ~4000 input (context) + 1000 output tokens

**Prompt Structure:**
```
System: You are a precise assistant. Answer only from the provided context.
        If the answer is not in the context, say "I don't know."
Context: {context_window}
Question: {rewritten_query}
Answer:
```

**Error Handling:**
- If LLM call fails → retry once with exponential backoff
- If second failure → return error response with trace_id

---

## 3.6 Validator Agent

**Responsibility:** Verify answer quality before returning to user.

**Token Budget:** ~2000 input + 300 output tokens

**Checks:**
1. **Groundedness:** Is every claim traceable to a retrieved chunk?
2. **Hallucination Detection:** Any facts absent from context?
3. **Completeness:** Does the answer address all parts of the question?

**Structured Output Schema:**
```json
{
  "groundedness":  0.82,
  "completeness":  0.74,
  "hallucination": 0.10,
  "final_score":   0.78
}
```

**Decision Logic:**
```
PASS    if final_score >= 0.75
RETRY   if final_score < 0.75 and retry_count < 2  → re-retrieve
DEGRADE if final_score < 0.75 and retry_count >= 2 → return with low_confidence flag
```

**Error Handling:**
- If validator LLM call fails → pass through (avoid blocking pipeline)
- Log validation skip with trace_id

---

## 3.7 Evaluator Agent

**Responsibility:** Log all metrics for offline analysis and dashboard.

**Token Budget:** N/A (writes to DB, no LLM call)

**Logs:**
- Retrieval: Precision@K, Recall@K, latency_ms
- Generation: groundedness_score, completeness_score, hallucination_score, final_score
- System: total_latency_ms, cost_estimate, retry_count
- Stage-level: `latency_breakdown` from RAGState (query_agent, retrieval, reranker, context_assembly, llm_answer, validator)
- LangSmith: full trace push

**Error Handling:**
- Non-blocking — log failures silently to avoid degrading user response

---

# 4. Ingestion Pipeline

## 4.1 Overview

```
Raw Document (PDF / HTML / Image / DOCX)
 ↓
Document Parser
    ├─ Extract text blocks
    ├─ Extract images → save to assets table
    └─ Extract tables → save to assets table
 ↓
Chunking Engine (Section 5)
 ↓
Embedding Service (Section 6)
    └─ Batch embed chunks (BGE, batch_size=64)
 ↓
Postgres Storage
    ├─ INSERT INTO documents
    ├─ INSERT INTO sections
    ├─ INSERT INTO chunks (with embedding + tsv)
    ├─ INSERT INTO assets
    └─ INSERT INTO relationships (chunk ↔ asset links)
 ↓
Index Update (ivfflat auto-updates on INSERT)
```

---

## 4.2 Celery Task Design

Ingestion is always async — never on the request path.

```python
@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_document(self, doc_path: str, metadata: dict):
    try:
        parsed    = parse_document(doc_path)
        chunks    = chunk_document(parsed)
        embeddings = embed_batch(chunks)
        store(chunks, embeddings, metadata)
    except Exception as exc:
        raise self.retry(exc=exc)
```

**Queue design:**
- `ingestion` queue: document parsing + chunking
- `embedding` queue: batch embedding (GPU-bound, separate worker pool)
- `storage` queue: DB writes

---

## 4.3 Error Handling

| Failure Point | Behavior |
|---|---|
| Parse failure | Mark document as `failed`, log error, do not retry automatically |
| Embedding API/model failure | Retry up to 3× with backoff |
| DB write failure | Retry up to 3×; on final failure, dead-letter queue |
| Partial ingestion | Transactional: roll back all tables for that doc_id |

---

# 5. Context-Aware Chunking Engine

## 5.1 Core Objective
Maximize semantic coherence while preserving structural and multimodal context.

---

## 5.2 Algorithm

### Step 1: Structural Segmentation
Split document into:
- headings
- paragraphs
- tables
- images

---

### Step 2: Semantic Boundary Detection

Uses the same `BAAI/bge-large-en-v1.5` model as document and query embeddings to ensure threshold calibration is consistent.

```
similarity_threshold = 0.75  # cosine similarity — tunable per domain via CHUNKING_SIMILARITY_THRESHOLD

similarity = cosine(emb(block_i), emb(block_i+1))

IF similarity < similarity_threshold:
    create_new_chunk()
ELSE:
    merge()
```

---

### Step 3: Context Window Injection

Each chunk contains:
- current text
- previous chunk summary
- section title
- document summary

---

### Step 4: Multimodal Linking Logic

```
IF image exists near text:
    link(image_id ↔ chunk_id)

IF table belongs to section:
    attach(table_id)
```

---

### Step 5: Chunk Size Optimization

Constraints:
- min_tokens = 150
- max_tokens = 600

```
IF chunk > max_tokens:
    split semantically

IF chunk < min_tokens:
    merge with adjacent
```

---

### Step 6: Metadata Enrichment

Each chunk stores:

```json
{
  "section_title": "...",
  "doc_id": "uuid",
  "modality": "text | image | table",
  "semantic_tags": ["tag1", "tag2"],
  "linked_elements": ["asset_id_1", "asset_id_2"],
  "chunk_summary": "...",
  "prev_chunk_summary": "..."
}
```

---

# 6. Embedding Service Design

## 6.1 Interface

```python
class Embedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
```

## 6.2 Implementations

| Provider | Class | Model | Dims |
|---|---|---|---|
| BGE (local) | `BGEEmbedder` | `BAAI/bge-large-en-v1.5` ✓ default | 1024 |
| OpenAI | `OpenAIEmbedder` | `text-embedding-3-large` | 3072 |

Selected via `EMBEDDING_PROVIDER = bge | openai` environment variable.

## 6.3 Requirements

- **Batch processing:** batch size 32–128 (configurable via `EMBEDDING_BATCH_SIZE`)
- **Normalization:** L2-normalize all embeddings for cosine similarity correctness
- **Retry logic:** exponential backoff on API/model failure (max 3 retries)
- **Optional caching:** cache embeddings by content hash in Redis to avoid re-embedding identical content

## 6.4 Embedding Consistency Constraint

⚠️ Violating this causes silent, severe retrieval degradation.

- Query embeddings and document embeddings MUST use the **same model**
- Must apply the **same normalization strategy** at both ingestion and query time
- The chunking engine (Section 5, Step 2) also uses this same model for semantic boundary detection
- Track the model version in the database per chunk (see Section 13 schema)

```bash
EMBEDDING_MODEL_VERSION=bge-large-en-v1.5
```

- At query time, assert: `query_embedding_model == document_embedding_model`; abort with error if mismatch.

---

# 7. PostgreSQL + pgvector Setup Strategy

## 7.1 Installation

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 7.2 Table Design

See Section 13 for full production schema.

---

## 7.3 Indexing Strategy

### Vector Index

```sql
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

See Section 16 for IVFFlat `lists` sizing guidance per corpus scale.

### Hybrid Search

- Vector similarity (ANN via ivfflat)
- BM25 via `tsvector` + GIN index

---

## 7.4 Query Plan

```sql
SELECT * FROM chunks
ORDER BY embedding <=> query_embedding
LIMIT K;
```

---

# 8. Retrieval Optimization Strategy

## 8.1 Query Expansion

- synonym expansion
- intent classification
- multi-query generation (3–5 variants)

---

## 8.2 Hybrid Fusion

Raw score fusion without normalization is incorrect — dense and BM25 scores operate on different scales.

**Correct approach:**

```
dense_norm = (dense_score - min(dense)) / (max(dense) - min(dense))
bm25_norm  = (bm25_score  - min(bm25))  / (max(bm25)  - min(bm25))

final_score = α * dense_norm + β * bm25_norm
```

Note: pgvector `<=>` returns cosine **distance** (lower = more similar), so invert before normalizing:
`dense_norm = 1 - normalized_distance`

Default: α = 0.7, β = 0.3 (tunable per domain via `DENSE_WEIGHT` / `SPARSE_WEIGHT` env vars)

---

## 8.3 Re-ranking

- Cross-encoder scoring
- Semantic relevance boost
- Re-rank Top-20 → return Top-5

---

# 9. Evaluation System Design

## 9.1 Offline Evaluation

Dataset:

```
(query, relevant_chunks, answer)
```

Metrics:
- Precision@K
- Recall@K
- nDCG

---

## 9.2 Online Evaluation

### Signals

- click-through
- user feedback (thumbs up/down)
- latency per component (p50, p95, p99)

---

## 9.3 LLM-as-Judge

```
score = LLM(answer, context) → structured {groundedness, completeness, hallucination, final_score}
```

Criteria:
- groundedness
- completeness
- factual correctness

---

# 10. RAG Evaluation Dashboard Design

## Panels

1. Retrieval Metrics
   - Precision@K, Recall@K, nDCG

2. Generation Metrics
   - Hallucination rate
   - Answer relevance
   - Completeness score

3. System Metrics
   - Latency (p50, p95, p99) — total and per stage
   - Cost per query
   - Retry rate

4. Agent Health
   - Per-agent success rate
   - Per-agent avg latency

---

## Tools

- LangSmith (tracing + eval)
- Custom dashboard (Streamlit / Superset)

---

# 11. Deployment Architecture

## 11.1 Services

- API Layer: FastAPI (async, Uvicorn/Gunicorn)
- Worker Layer: Celery + Redis (async ingestion jobs)
- DB Layer: Postgres + pgvector
- Cache: Redis (query cache, frequent-query dedup)
- Tracing: LangSmith

---

## 11.2 Environment & Secrets

All secrets are injected via environment variables — never hardcoded.

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/rag_db

# LLM
ANTHROPIC_API_KEY=sk-...         # or OPENAI_API_KEY
LLM_MODEL=claude-sonnet-4-6      # configurable per environment

# Embeddings
EMBEDDING_PROVIDER=bge                        # bge | openai
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5       # local model, no API key required
EMBEDDING_MODEL_VERSION=bge-large-en-v1.5
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=64

# Chunking
CHUNKING_SIMILARITY_THRESHOLD=0.75

# Retrieval
TOP_K_PRE_RERANK=20
TOP_K_POST_RERANK=5
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3

# Validation
VALIDATOR_THRESHOLD=0.75
MAX_RETRIES=2

# Observability
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=multimodal-rag

# Redis
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600
```

### Secrets Management
- **Local dev:** `.env` file (excluded from git via `.gitignore`); copy from `.env.example`
- **Production:** AWS Secrets Manager / GCP Secret Manager / Vault
- Never log secret values; mask in traces

---

## 11.3 Scaling Strategy

### Concurrency Model

```
FastAPI (async) + Uvicorn workers = CPU cores * 2
Celery workers for ingestion = separate pool, auto-scaled
Celery workers for embedding = GPU pool (if local BGE)
Connection pool: asyncpg pool size = 10–20 per worker
```

### Horizontal Scaling

- Retrieval service is stateless → scale horizontally behind a load balancer
- Sticky sessions NOT required (all state in Postgres/Redis)
- Embedding generation: offload to batch Celery tasks, not request path

### Cache Invalidation

```python
# Query cache key = hash(rewritten_query + metadata_filters)
# TTL = 1 hour (configurable via CACHE_TTL_SECONDS)
# Invalidate on: new document ingested into same domain
cache_key = sha256(f"{query}:{filters}").hexdigest()
```

### Load Balancing

- Use round-robin for stateless API replicas
- Postgres read replicas for retrieval queries (write → primary, read → replica)
- Redis cluster for cache HA

---

## 11.4 Health Checks

### API Health Endpoint

```json
GET /health
→ {
    "status": "ok | degraded | down",
    "db": "ok | error",
    "redis": "ok | error",
    "llm": "ok | error",
    "timestamp": "2024-01-01T00:00:00Z"
  }
```

### Liveness vs Readiness

- **Liveness** (`/health/live`): process is alive (always fast)
- **Readiness** (`/health/ready`): DB + Redis + LLM reachable (used by k8s)

### Component Timeouts

| Component        | Timeout |
|------------------|---------|
| Dense retrieval  | 3s      |
| Sparse retrieval | 2s      |
| Reranker         | 2s      |
| Context assembly | 1s      |
| LLM (answer)     | 30s     |
| LLM (validator)  | 15s     |
| Total pipeline   | 60s     |

---

## 11.5 Observability

- **Tracing:** LangSmith (full agent traces with token counts and latencies)
- **Logs:** Structured JSON logs → stdout → log aggregator (Datadog / CloudWatch)
- **Metrics:** Prometheus-compatible `/metrics` endpoint
  - `rag_query_latency_seconds` (histogram, total)
  - `rag_stage_latency_seconds{stage="query_agent|retrieval|reranker|context_assembly|llm_answer|validator"}` (histogram, per stage)
  - `rag_retrieval_chunks_returned` (gauge)
  - `rag_validator_pass_rate` (gauge)
  - `rag_retry_total` (counter)
  - `rag_zero_results_total` (counter)

---

# 12. Production Enhancements

- Caching frequent queries (Redis, TTL-based)
- Adaptive retrieval (dynamic K based on query confidence)
- User personalization (per-user retrieval filters)
- Incremental indexing (new docs appended without full reindex)
- A/B testing retrieval strategies (α/β weights, Top-K values)

---

# 13. Production-Grade PostgreSQL Schema (Multimodal RAG)

## 13.1 Core Tables

### documents
```sql
CREATE TABLE documents (
    doc_id      UUID PRIMARY KEY,
    title       TEXT,
    source      TEXT,
    doc_type    TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    metadata    JSONB
);
```

### sections
```sql
CREATE TABLE sections (
    section_id          UUID PRIMARY KEY,
    doc_id              UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_section_id   UUID,
    title               TEXT,
    level               INT,
    metadata            JSONB
);
```

### chunks
```sql
CREATE TABLE chunks (
    chunk_id               UUID PRIMARY KEY,
    doc_id                 UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id             UUID REFERENCES sections(section_id),
    content                TEXT,
    embedding              VECTOR(1024),
    embedding_model_version TEXT DEFAULT 'bge-large-en-v1.5',
    modality               TEXT,
    token_count            INT,
    metadata               JSONB,
    tsv                    TSVECTOR,
    created_at             TIMESTAMP DEFAULT NOW()
);
```

### assets (images/tables)
```sql
CREATE TABLE assets (
    asset_id    UUID PRIMARY KEY,
    doc_id      UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    type        TEXT,
    content     TEXT,
    embedding   VECTOR(1024),
    metadata    JSONB
);
```

### relationships
```sql
CREATE TABLE relationships (
    id              SERIAL PRIMARY KEY,
    source_id       UUID,
    target_id       UUID,
    relation_type   TEXT
);
```

---

## 13.2 Indexes

```sql
-- Vector similarity search
CREATE INDEX idx_chunks_embedding
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- BM25 full-text search
CREATE INDEX idx_chunks_tsv ON chunks USING GIN(tsv);

-- FK lookups
CREATE INDEX idx_chunks_doc_id     ON chunks (doc_id);
CREATE INDEX idx_chunks_section_id ON chunks (section_id);
CREATE INDEX idx_assets_doc_id     ON assets (doc_id);

-- Relationship lookups in both directions
CREATE INDEX idx_relationships_src ON relationships (source_id);
CREATE INDEX idx_relationships_tgt ON relationships (target_id);
```

---

## 13.3 BM25 Trigger

```sql
CREATE OR REPLACE FUNCTION update_tsv() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tsv_update
BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION update_tsv();
```

---

## 13.4 Evaluation Logs

```sql
CREATE TABLE retrieval_logs (
    id                  SERIAL PRIMARY KEY,
    trace_id            UUID,
    query               TEXT,
    retrieved_chunk_ids UUID[],
    scores              FLOAT[],
    latency_ms          INT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE evaluation_logs (
    id                  SERIAL PRIMARY KEY,
    trace_id            UUID,
    query               TEXT,
    answer              TEXT,
    groundedness_score  FLOAT,
    completeness_score  FLOAT,
    hallucination_score FLOAT,
    final_score         FLOAT,
    retry_count         INT,
    confidence          FLOAT,
    latency_breakdown   JSONB,
    feedback            TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);
```

---

# 14. Query Patterns

### Hybrid Retrieval (Normalized Scores)
```sql
WITH scores AS (
    SELECT chunk_id, content, metadata,
           (embedding <=> :query_embedding) AS vector_dist,
           ts_rank(tsv, plainto_tsquery(:query)) AS bm25_score
    FROM chunks
    WHERE metadata->>'domain' = 'security'
),
bounds AS (
    SELECT min(vector_dist) AS v_min, max(vector_dist) AS v_max,
           min(bm25_score)  AS b_min, max(bm25_score)  AS b_max
    FROM scores
)
SELECT s.*,
    (  0.7 * (1 - (s.vector_dist  - b.v_min) / NULLIF(b.v_max - b.v_min, 0))
     + 0.3 * (    (s.bm25_score   - b.b_min) / NULLIF(b.b_max - b.b_min, 0))
    ) AS final_score
FROM scores s, bounds b
ORDER BY final_score DESC
LIMIT 10;
```

### Linked Context Fetch
```sql
SELECT c2.*
FROM relationships r
JOIN chunks c2 ON r.target_id = c2.chunk_id
WHERE r.source_id = :chunk_id;
```

### Section Context Expansion
```sql
SELECT *
FROM chunks
WHERE section_id = :section_id
ORDER BY created_at;
```

---

# 15. Local & Self-Hosted Setup

## 15.1 Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: ankane/pgvector
    container_name: pgvector-db
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin
      POSTGRES_DB: rag_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d rag_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  api:
    build: .
    container_name: rag-api
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  worker:
    build: .
    container_name: rag-worker
    command: celery -A ingestion.tasks worker --loglevel=info -Q ingestion,embedding,storage
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  pgdata:
```

---

## 15.2 Setup Steps

```bash
# Copy env template and fill in secrets
cp .env.example .env

# Start all services (API + worker + DB + Redis)
docker-compose up -d

# Connect to Postgres
psql -h localhost -U admin -d rag_db

# Run migrations
python -m db.migrations.run
```

Enable extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

# 16. Scaling Strategy

## Phase 1 — Single Instance
- Single API + single Postgres
- IVFFlat `lists = 100` (suitable for up to ~100K chunks)
- Redis on same host

## Phase 2 — Scale Out
- Tune IVFFlat lists: use `lists = sqrt(num_rows)` as a baseline; set `ivfflat.probes = lists / 10` at query time for recall vs. speed trade-off
- Partition chunks by doc_id
- Add Postgres read replica (retrieval reads → replica)
- Celery workers for ingestion + embedding jobs (separate pools)

## Phase 3 — Distributed
- Split architecture:
  - Postgres → metadata + relationships
  - Dedicated vector DB (Pinecone / Weaviate / Qdrant) → embeddings
- API: Kubernetes deployment with HPA (CPU + latency metrics)
- Redis Cluster for cache HA

### IVFFlat `lists` Sizing Reference

| Corpus Size | Recommended `lists` |
|---|---|
| < 100K chunks | 100 |
| 100K – 1M chunks | `sqrt(num_rows)` (~316–1000) |
| > 1M chunks | Migrate to HNSW or dedicated vector DB |

---

# 17. Common Pitfalls

| Pitfall | Impact | Fix |
|---|---|---|
| Missing relationships table | No multimodal linkage | Always create relationships at ingestion |
| No BM25 | Poor recall for keyword queries | Enable tsvector + GIN index |
| No evaluation logs | No improvement loop | Evaluator agent must always write |
| Oversized chunks | Weak retrieval signal | Enforce 150–600 token constraint |
| No retry cap | Infinite loops | Set MAX_RETRIES=2 in state |
| Secrets in code | Security breach | Use .env + secrets manager |
| No health checks | Silent failures in prod | Implement /health/live + /health/ready |
| Synchronous ingestion | Blocks API | Use Celery async workers |
| Unnormalized hybrid scores | Dense dominates; BM25 ignored | Min-max normalize each score set before fusion |
| Embedding model mismatch | Silent severe retrieval degradation | Enforce same model+version at ingest and query time |
| Wrong VECTOR dim in schema | Embeddings fail to insert | VECTOR(1024) for bge-large-en-v1.5 |
| No stage-level latency | Can't identify bottlenecks | Track latency_breakdown per stage in RAGState |
| Missing target_id index | Slow context assembly joins | Add idx_relationships_tgt on relationships(target_id) |
| IVFFlat lists undersized | Poor ANN recall at scale | Tune lists = sqrt(num_rows); re-index on growth |
| Wrong embedder in chunking | Threshold calibrated against wrong space | Use same BGE model for boundary detection |
| No ingestion rollback | Partial doc state in DB | Wrap per-document inserts in a transaction |

---

# 18. Final Capabilities

✔ Multimodal ingestion pipeline (Celery async, transactional, error-resilient)
✔ Context-aware chunking with explicit similarity threshold (0.75, BGE-calibrated)
✔ Hybrid retrieval with normalized score fusion (dense + sparse + metadata filter)
✔ Embedding abstraction layer (BGE-large local / OpenAI API, pluggable)
✔ Embedding consistency enforcement (model versioning per chunk, query-time assertion)
✔ Context assembly pipeline (section expansion + asset linking + dedup + truncation)
✔ Agentic orchestration (LangGraph with typed state, context_assembly node)
✔ Self-healing retrieval loop (validator → retry → degrade) with structured validator scores
✔ Explicit retrieval failure fallback (zero-results → evaluator, no silent drop)
✔ Full evaluation pipeline (offline + online + LLM-as-judge)
✔ Stage-level latency tracking (per-agent breakdown in state + Prometheus histogram)
✔ Production-grade storage design (Postgres + pgvector, VECTOR(1024), bidirectional indexes)
✔ Observability (LangSmith tracing + Prometheus metrics + structured logs)
✔ Secrets management and environment configuration
✔ Horizontal scaling with load balancing, IVFFlat tuning guidance, and caching

---

**End of Extended Design Document**
