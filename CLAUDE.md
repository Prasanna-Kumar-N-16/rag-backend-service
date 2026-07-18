# CLAUDE.md

Guidance for working in this repository.

## What this is

Production-oriented Retrieval-Augmented Generation backend: **FastAPI +
PostgreSQL/pgvector + Anthropic Claude**, with hybrid retrieval, Cohere
reranking, and Responsible-AI guardrails. Python 3.12, strict typing.

## Commands

```bash
pip install -e ".[dev]"     # install with dev tooling
ruff check .                # lint  (must be clean)
mypy app                    # type-check, strict  (must be clean)
pytest -q                   # tests — fully offline, no DB/keys needed
uvicorn app.main:app --reload   # run locally
docker compose up --build       # run full stack (API + pgvector)
python -m app.evaluation --dataset data/golden_dataset.example.jsonl  # eval
```

CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest on push/PR. Keep all
three green; the same three commands are the definition of "done".

## Layer map (`app/`)

| Package | Responsibility |
|---------|----------------|
| `api` | FastAPI routes, middleware, schemas, dependency providers |
| `indexing` | S3 load → chunk → Voyage embed → pgvector upsert |
| `retrieval` | vector + lexical search, RRF fusion, Cohere rerank |
| `generation` | grounded prompt assembly, Claude synthesis (streamed + retry) |
| `guardrails` | PII masking, prompt-injection defense, output validation |
| `evaluation` | golden-dataset retrieval + faithfulness metrics + CLI |

`/v1/query` pipeline order: guard-in (query injection → PII-mask-for-logs) →
hybrid retrieve → rerank → guard-in (context injection) → synthesize →
guard-out (groundedness + PII re-mask). See `app/api/routes/query.py`.

## Conventions & invariants (read before editing)

- **Config is env-driven** via `pydantic-settings` (`app/config.py`); never
  hardcode models/keys/tunables. Provider keys are optional so the app boots
  health-only. `get_settings()` is `lru_cache`d — one instance per process.
- **Tests run offline.** `tests/conftest.py` mocks the DB lifecycle and all
  provider clients. Do not introduce tests that need a live Postgres or real
  API keys. Prefer injecting fakes (e.g. the evaluator takes callables).
- **asyncpg: one operation per connection.** Never `asyncio.gather(...)` over
  multiple `conn.execute(...)` on a single acquired connection — it raises
  `InterfaceError`. Batch with `executemany`, or acquire a connection per op.
  (See `app/indexing/indexer.py`.)
- **pgvector bootstrap ordering.** The `vector` extension must exist before the
  pgvector codec is registered; `app/db.py:_init_connection` creates the
  extension in the pool initializer for exactly this reason. Don't move codec
  registration ahead of extension creation.
- **Shared service singletons** (`app/api/dependencies.py`) use double-checked
  locking because FastAPI runs sync dependencies in a threadpool. Keep the lock
  when adding new singletons; expose them via `Depends()` for test overrides.
- **Guardrails are policy-free detectors.** They return structured results
  (risk level, matched patterns); the *route* decides to block/log/allow. The
  query route blocks on `risk_level == "high"`.
- **Style:** strict mypy (`disallow_untyped_defs`), ruff rules
  `E,F,I,UP,B,C4,SIM`, line length 100. Frozen dataclasses for data passed
  between pipeline stages. Structured logging via `app.logging.get_logger`.

## Commits

Small, single-purpose commits with a conventional-commit prefix
(`fix:` / `feat:` / `ci:` / `build:`), matching the existing history. Run the
three gate commands before committing.
```
