# Security Policy

`rag-backend-service` processes untrusted input (user queries and
retrieved documents) and ships Responsible-AI guardrails — PII masking,
prompt-injection defense, and output validation. We take security reports
seriously.

## Supported versions

The project is pre-1.0 and under active development. Security fixes are applied
to the `master` branch. Until a stable release line exists, only `master` is
supported.

| Version | Supported |
| ------- | --------- |
| `master` (latest) | ✅ |
| tagged pre-1.0 releases | ⚠️ best-effort |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report privately via one of:

- **GitHub Security Advisories** — use the repository's
  *Security → Report a vulnerability* tab (preferred; enables coordinated
  disclosure).
- **Email** — send details to the maintainer at `radhaprasanna77@gmail.com`
  with `[SECURITY]` in the subject line.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal proof-of-concept if possible).
- Affected component/file and any relevant configuration.

## What to expect

- **Acknowledgement** within 3 business days.
- An initial assessment and severity triage within 7 business days.
- Coordinated disclosure: we will agree on a timeline before any public
  disclosure and credit you (if you wish) in the release notes.

## Scope & hardening notes

This project defends against, among others:

- **Direct and indirect prompt injection** — user queries and retrieved
  context chunks are scanned (`app/guardrails/prompt_injection.py`).
- **PII leakage** — inputs and outputs are masked
  (`app/guardrails/pii.py`, Presidio or regex fallback).
- **Ungrounded / malformed output** — responses are validated
  (`app/guardrails/output_guard.py`).

Reports that improve or bypass these controls are especially welcome.
Never test against systems or data you do not own or have explicit
permission to test.
