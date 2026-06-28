# rag-backend-service

Production-ready Retrieval-Augmented Generation (RAG) backend.

**Stack:** Python 3.12 · FastAPI · PostgreSQL + pgvector · Anthropic Claude
(`claude-opus-4-8`) · Voyage AI embeddings · Cohere reranking · Docker · Kubernetes.

> 🚧 **Status: Day 1 of a 7-day build.** The project skeleton, configuration,
> structured logging, and health/readiness endpoints are in place. Ingestion,
> retrieval, generation, guardrails, evaluation, and infra land on later days
> (see the delivery schedule in the plan).

## Architecture (target)

| Layer | Responsibility |
|-------|----------------|
| `app/api` | FastAPI routes, middleware, schemas, dependencies |
| `app/indexing` | S3 loading · chunking · Voyage embeddings · pgvector upsert |
| `app/retrieval` | Hybrid search (vector + lexical, RRF) · Cohere reranking |
| `app/generation` | Grounded prompt assembly · Claude synthesis |
| `app/guardrails` | PII masking · prompt-injection defense · output validation |
| `app/evaluation` | Golden-dataset faithfulness evaluation |

## Quickstart

```bash
# 1. Create a virtual environment and install (dev extras included)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure (all keys optional for the health-only Day 1 boot)
cp .env.example .env

# 3. Run
uvicorn app.main:app --reload

# 4. Verify
curl localhost:8000/healthz
curl localhost:8000/readyz
```

## Development

```bash
ruff check .        # lint
mypy app            # type-check (strict)
pytest -q           # tests
```

Interactive API docs are served at `http://localhost:8000/docs` while running.

## Configuration

All settings are environment-driven (`app/config.py`, `pydantic-settings`).
See [`.env.example`](.env.example) for the full list — provider keys, model
IDs, retrieval tunables, and the pgvector connection string.
