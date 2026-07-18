# rag-backend-service

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-115-brightgreen.svg)](tests/)
[![Type-checked: mypy strict](https://img.shields.io/badge/mypy-strict-blue.svg)](pyproject.toml)

Production-ready Retrieval-Augmented Generation (RAG) backend.

**Stack:** Python 3.12 · FastAPI · PostgreSQL + pgvector · Anthropic Claude
(`claude-opus-4-8`) · Voyage AI embeddings · Cohere reranking.

> ✅ **Status:** FastAPI scaffold, config & structured logging,
> health/readiness probes, the pgvector data layer & ingestion pipeline (S3 →
> chunk → embed → upsert), multi-stage hybrid retrieval (vector + lexical +
> RRF + Cohere rerank), grounded Claude generation with the API layer,
> Responsible-AI guardrails (PII masking, prompt-injection defense, output
> validation), a golden-dataset evaluation harness (retrieval + faithfulness
> metrics + CLI), and a container stack (Dockerfile + `docker compose` app +
> pgvector). **115 tests** pass under strict `mypy`, enforced in CI. Durable
> ingestion (replacing the in-process `BackgroundTasks` with a real worker
> queue) is the remaining milestone.

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

# 2. Configure (all keys optional for the health-only boot)
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

## Run with Docker

The full stack (API + PostgreSQL/pgvector) comes up with one command:

```bash
# Provider keys are read from your shell env (or an .env file Compose loads).
export ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... COHERE_API_KEY=... API_KEY=...
docker compose up --build

curl localhost:8000/healthz   # liveness
curl localhost:8000/readyz    # readiness (DB reachable)
curl -H "X-API-Key: $API_KEY" -d '{"query": "..."}' localhost:8000/v1/query
```

The API boots health-only if no provider keys are set; `/v1/query` and
`/v1/ingest` need them, plus `API_KEY` — both routes require a matching
`X-API-Key` header on every request and refuse to boot unauthenticated (500)
if `API_KEY` isn't configured. Database schema (extension, table, HNSW + GIN
indexes) is bootstrapped automatically on first boot.

## Evaluation

Score the live pipeline against a golden dataset (retrieval recall/MRR,
answer similarity, and faithfulness):

```bash
python -m app.evaluation --dataset data/golden_dataset.example.jsonl
# add --min-faithfulness 0.5 to exit non-zero below a threshold (CI gate)
```

The metric functions in `app/evaluation/metrics.py` are pure and offline-tested;
the evaluator accepts injectable retrieve/synthesize callables so it can run
against the real stack or against fakes.

## Configuration

All settings are environment-driven (`app/config.py`, `pydantic-settings`).
See [`.env.example`](.env.example) for the full list — provider keys, model
IDs, retrieval tunables, and the pgvector connection string.

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for
the development setup, coding standards (ruff + strict mypy + pytest), and the
pull-request workflow. By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Found a vulnerability? Please **do not** open a public issue — follow the
responsible-disclosure process in [`SECURITY.md`](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
