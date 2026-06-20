# Contributing to Respawn

Thanks for your interest. This document describes the priority roadmap and the one hard rule.

---

## The one hard rule

**Don't break the guardrails.**

The `tests/` suite is the contract. Every PR must pass `make test` on Python 3.9–3.12. If you need to change a test to make your code pass, that's a sign the abstraction is wrong — fix the abstraction.

---

## Roadmap (priority order)

These are the things most likely to be merged quickly, roughly in order of value:

1. **Cascading failure support** — `recover.py` currently handles single failures only. The big win is chaining recovery plans when re-entry itself triggers a second failure.

2. **LLM-as-judge scoring in RecoveryBench** — TCR is currently a binary heuristic. Replace with an LLM judge that scores partial completions.

3. **Temporal / Restate adapters** — `core.py` is framework-agnostic but has no first-class connectors. A thin adapter for each major durable-execution runtime would unlock the quickstart story.

4. **Streaming trace support** — `recover.py` currently requires a complete trace. Online (streaming) attribution is theoretically possible with Who&When but not yet wired up.

5. **More archetypes in RecoveryBench** — `critic` and `orchestrator` archetypes are thin. More diverse scenarios improve benchmark coverage.

6. **Benchmark leaderboard** — a static page in `docs/` tracking results across model versions.

---

## Getting started

```bash
git clone https://github.com/harshittaneja/respawn
cd respawn
pip install -e ".[dev]"
make test
```

---

## Before opening a PR

- Open an issue first for anything non-trivial. Saves everyone time.
- Keep PRs focused. One logical change per PR.
- Add or update tests. PRs without tests for new behaviour will be asked to add them.
- Run `make lint` and `make test` locally before pushing.

---

## Commit style

```
<type>: <short summary>

<optional body>
```

Types: `feat`, `fix`, `bench`, `docs`, `test`, `refactor`, `chore`.

---

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
