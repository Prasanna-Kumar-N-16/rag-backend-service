# Guardrail fixes & CI

Three bugs turned up during code review, plus a CI workflow to catch this class of regression going forward.

## Presidio engines were rebuilt on every call

`app/guardrails/pii.py`'s `_presidio_pass` constructed a fresh `AnalyzerEngine()` and `AnonymizerEngine()` on every invocation. `mask_pii()` — which calls it — runs on every `/v1/query` request, both for the query guard-in and the output guard-out. With the optional `presidio` extra installed, `AnalyzerEngine()` loads a spaCy NLP model on construction, so this reloaded a full model on every single request.

**Fix:** cache the engine pair for the process lifetime with `functools.lru_cache`. A `None` sentinel already had to represent "presidio isn't installed," so the cache slots in cleanly around that.

## Retrieved context wasn't scanned for prompt injection

`/v1/query` only ran `check_injection` against the user's own query text. A poisoned source document ingested via `/v1/ingest` could carry injection text (e.g. "Ignore previous instructions...") that flows straight into the Claude prompt through the reranked context chunks — the classic indirect/second-order injection vector for RAG systems — and it sailed straight past the guard.

**Fix:** added `check_context_injection`, which runs the same detector over the concatenated chunk text after reranking and before synthesis, blocking with a 400 when `risk_level == "high"` — mirroring the existing query-side check.

## Singleton construction had a race under concurrent requests

`app/api/dependencies.py`'s service singletons (`Embedder`, `Reranker`, `ClaudeClient`, `Synthesizer`) used a bare check-then-assign (`if _x is None: _x = X(settings)`). FastAPI runs sync dependencies in a thread pool, so concurrent first requests could both pass the `None` check, both construct an instance, and the last write wins — silently leaking the discarded instance's HTTP client.

**Fix:** double-checked locking around construction, using a shared `threading.Lock` for all four singletons.

## CI

`ruff`, `mypy --strict`, and `pytest` were already configured but never enforced. Added `.github/workflows/ci.yml` to run all three on every push and pull request — the same commands used locally, now gating merges instead of just being documented.
