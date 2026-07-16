# Contributing to rag-backend-service

Thanks for your interest in contributing! This document explains how to set up
your environment, the standards we hold code to, and how to get a change merged.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Requires **Python 3.12**.

```bash
# Create a virtual environment and install with dev extras
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# (Optional) install the heavyweight Presidio PII backend
pip install -e ".[dev,pii]"

# Configure — all provider keys are optional for the offline test suite
cp .env.example .env
```

## Quality gates

Every change must pass all three checks locally before you open a PR. CI runs
the identical commands, so green locally means green in CI.

```bash
ruff check .        # lint  (also: `ruff format .` to auto-format)
mypy app            # strict type-checking — no new ignores without justification
pytest -q           # full test suite (offline; no live Postgres or API keys needed)
```

The test suite runs entirely offline — Postgres, S3, and provider SDKs are
mocked (see [`tests/conftest.py`](tests/conftest.py)). You do **not** need real
credentials to develop or run tests.

## Coding standards

- **Typed, strictly.** `mypy --strict` is enforced; annotate all new functions.
- **Line length 100**, import sorting and lint rules per [`pyproject.toml`](pyproject.toml)
  (`E, F, I, UP, B, C4, SIM`).
- **Pinned dependencies.** We pin exact versions (`==`) for reproducibility.
  Bump deliberately, in their own commit, with a note on why.
- **Match the surrounding code.** Mirror existing naming, module layout, and
  comment density. Each subsystem lives under its own `app/` package
  (`api`, `indexing`, `retrieval`, `generation`, `guardrails`, `evaluation`).
- **Tests first.** New behavior needs a test; bug fixes need a regression test
  that fails before the fix.

## Commit & PR workflow

1. Branch from `master`. Use a descriptive name, e.g. `feat/<topic>`,
   `fix/<topic>`, or `docs/<topic>`.
2. Write focused commits with clear messages (imperative mood, e.g.
   `fix: cache Presidio engines instead of rebuilding per call`).
3. Ensure `ruff`, `mypy`, and `pytest` all pass.
4. Open a PR against `master` and fill in the PR template. Link any related
   issue.
5. A maintainer reviews; address feedback by pushing follow-up commits.

## Reporting bugs & requesting features

Use the [issue templates](.github/ISSUE_TEMPLATE/). For **security**
vulnerabilities, do **not** open a public issue — follow
[`SECURITY.md`](SECURITY.md) instead.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE) that covers this project.
