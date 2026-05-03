# Multimodal Agentic RAG

A production-grade Retrieval-Augmented Generation system that handles text, images, and tables. Built with a LangGraph agentic pipeline, hybrid retrieval (dense + sparse), and a self-healing validation loop.

---

## What it does

- **Ingests** PDF, HTML, DOCX, and image files — captions images via vision LLM (Claude / GPT-4o)
- **Chunks** documents using semantic boundary detection (BGE embeddings, cosine similarity)
- **Retrieves** with hybrid search: pgvector ANN + BM25, normalized fusion, cross-encoder reranking
- **Answers** questions using a LangGraph agent pipeline with groundedness validation and retry loops
- **Evaluates** answer quality offline (Precision@K, nDCG) and online (LLM-as-judge)

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| LLM | Claude (Anthropic) |
| Embeddings | `BAAI/bge-large-en-v1.5` (local) or OpenAI |
| Vision | Claude vision / GPT-4o (image captioning) |
| Vector DB | PostgreSQL + pgvector |
| Full-text search | PostgreSQL tsvector (BM25) |
| Cache | Redis |
| Async workers | Celery |
| Tracing | LangSmith |

---

## Project Structure

```
multimodal-agentic-rag/
│
├── data/
│
├── app/
│   ├── main.py
│   ├── dependencies.py
│   ├── routes/
│   └── schemas/
│
├── core/
│   ├── config.py
│   ├── logging.py
│   ├── constants.py
│   └── interfaces/
│       └── embedder.py
│
├── db/
│   ├── session.py
│   ├── schema.sql
│   ├── models/
│   ├── repositories/
│   └── migrations/
│
├── ingestion/
│   ├── parser.py
│   ├── tasks.py
│   └── pipeline.py
│
├── chunking/
│   └── engine.py
│
├── embeddings/
│   ├── bge.py
│   └── openai.py
│
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   ├── hybrid.py
│   └── reranker.py
│
├── agents/
│   ├── query_agent.py
│   ├── query_expander_agent.py
│   ├── retrieval_agent.py
│   ├── context_assembly.py
│   ├── answer_agent.py
│   ├── validator_agent.py
│   └── evaluator_agent.py
│
├── orchestration/
│   └── graph.py
│
├── evaluation/
│   ├── offline.py
│   ├── online.py
│   └── llm_judge.py
│
├── services/
│   └── cache.py
│
├── utils/
├── scripts/
├── tests/
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.12+
- Docker + Docker Compose
- [uv](https://github.com/astral-sh/uv) (package manager)

### 1. Clone and install dependencies

```powershell
git clone <repo-url>
cd multimodal-agentic-rag
uv sync
```

### 2. Configure environment

```powershell
copy .env.example .env
```

Edit `.env` and fill in at minimum:

```bash
ANTHROPIC_API_KEY=sk-ant-...     # Required for LLM agents + vision captioning
DATABASE_URL=postgresql+asyncpg://admin:admin@localhost:5432/rag_db
```

All other values have working defaults for local development.

### 3. Start infrastructure

```powershell
docker-compose up -d
docker-compose ps    # wait until both postgres and redis show healthy
```

### 4. Apply database schema

```powershell
python -m db.migrations.run
```

This creates all tables, indexes (ivfflat + GIN), and the BM25 trigger.

### 5. Install additional packages

```powershell
uv add redis pydantic-settings tabulate
```

---

## Ingesting Documents

```python
from ingestion.tasks import ingest_document_task

# Async via Celery worker
ingest_document_task.delay("data/my_document.pdf", {"domain": "finance"})
```

Or directly in Python (sync, for testing):

```python
import asyncio
from ingestion.pipeline import ingest_document
from db.session import get_session_factory

async def run():
    async with get_session_factory()() as session:
        doc_id = await ingest_document(
            path="data/my_document.pdf",
            metadata={"domain": "finance"},
            session=session,
        )
        await session.commit()
        print(f"Ingested: {doc_id}")

asyncio.run(run())
```

### Vision captioning

Images in documents are automatically captioned by the configured vision LLM:

```bash
VISION_PROVIDER=anthropic    # anthropic | openai | none
VISION_MODEL=claude-sonnet-4-6
```

Set `VISION_PROVIDER=none` to skip captioning (images stored with placeholder text).

### Start Celery worker

```powershell
celery -A ingestion.tasks worker --loglevel=info -Q ingestion,embedding,storage
```

> Use the default prefork pool. Do not use `--pool=gevent` or `--pool=eventlet` — incompatible with `asyncio.run()`.

---

## Configuration

All configuration is via environment variables. See [.env.example](.env.example) for the full reference.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `bge` | `bge` (local) or `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Must match at ingestion and query time |
| `EMBEDDING_DIM` | `1024` | Must match the model's output dimension |
| `CHUNKING_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity below which a new chunk is created |
| `CHUNK_MIN_TOKENS` | `150` | Minimum chunk size (merge if below) |
| `CHUNK_MAX_TOKENS` | `600` | Maximum chunk size (split if above) |
| `DENSE_WEIGHT` | `0.7` | Weight for vector score in hybrid fusion |
| `SPARSE_WEIGHT` | `0.3` | Weight for BM25 score — must sum to 1.0 with `DENSE_WEIGHT` |
| `VISION_PROVIDER` | auto | `anthropic`, `openai`, or `none` |
| `VALIDATOR_THRESHOLD` | `0.75` | Minimum score to pass answer validation |

---

## Architecture

The system follows an 8-phase development plan:

| Phase | Module | Status |
|---|---|---|
| 1 | Infrastructure & Configuration | ✅ Complete |
| 2 | Ingestion Pipeline | ✅ Complete |
| 3 | Retrieval System | 🔲 Planned |
| 4 | Agent Development | 🔲 Planned |
| 5 | Orchestration (LangGraph) | 🔲 Planned |
| 6 | API Layer | 🔲 Planned |
| 7 | Evaluation System | 🔲 Planned |
| 8 | Production Hardening | 🔲 Planned |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design and [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the phase-by-phase implementation plan.

---

## Development

### Run notebooks

Each phase has exploratory notebooks in `notebooks/`:

```
notebooks/
├── phase1_infrastructure/   # DB connection, schema setup, config validation
├── phase2_ingestion/        # Parsing, chunking, embedding, storage
└── ...
```

### Run migrations

```powershell
python -m db.migrations.run
```

### Project references

- [ARCHITECTURE.md](ARCHITECTURE.md) — full system design, schema, deployment
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) — phase-wise plan with file descriptions
- [.env.example](.env.example) — all environment variables with descriptions
