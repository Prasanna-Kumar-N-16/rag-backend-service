# syntax=docker/dockerfile:1

# ── Builder: install the package (+ pinned deps) into an isolated prefix ──────
FROM python:3.12-slim AS builder

WORKDIR /build
# pyproject needs README.md (readme) and LICENSE (license-files) at build time.
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install .

# ── Runtime: minimal image, non-root, installed package only ─────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Liveness check hits /healthz (process-up; does not require the DB).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
