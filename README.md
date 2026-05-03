### Project structure

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
│
├── scripts/
├── tests/
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
