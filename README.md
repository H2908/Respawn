# 🎮 respawn

<p align="center">
  <strong>Your agent failed at step 9. The real mistake was step 3.<br>
  Most tools retry step 9. respawn finds step 3.</strong>
</p>

<p align="center">
  <a href="https://github.com/H2908/respawn/actions/workflows/ci.yml"><img src="https://github.com/H2908/respawn/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome"></a>
</p>

---

## 😤 The Problem Everyone Ignores

You're running an AI agent pipeline. It fails.

What do you do? You **retry the step that crashed.**

That feels logical. But it's almost always wrong.

Here's why: we analyzed **184 real LLM multi-agent failures** (from the [Who&When](https://github.com/ag2ai/Agents_Failure_Attribution) dataset, ICML 2025) and found something shocking:

<p align="center">
  <br>
  <b>In 95.7% of failures, the real cause happened UPSTREAM — not at the step that crashed.</b><br>
  The median distance between cause and crash? <b>5 steps.</b>
  <br><br>
</p>

> **Translation:** Every retry library, every durable execution framework — they're all retrying the *wrong step*. They can structurally fix **at most 4.3%** of real agent failures.

![Who&When Results](docs/images/whoandwhen_results.png)

---

## 💡 The Insight

If the real mistake happened upstream, then the fix is simple in theory:

**Rewind to where it went wrong. Retry from there. Differently.**

But here's the honest part — it doesn't always help. We tested it on real models and found a clear boundary:

| Failure Type | Does rewinding help? | Why |
|---|---|---|
| 🧠 Reasoning trace errors | ❌ No | A smart model re-reads the full transcript and self-corrects anyway |
| ⚙️ State-corrupting pipeline faults | ✅ Yes — significantly | Once state is corrupted, re-reading can't fix it. You must redo the step. |

**Real numbers from our RecoveryLab experiment (n=150, Claude Haiku):**
- With respawn: **93% recovery**
- Without respawn: **79% recovery**
- Statistical significance: McNemar **p ≈ 0.001**

![RecoveryLab Results](docs/images/recoverylab_results.png)

The contribution isn't a magic bullet. It's knowing **exactly when rewinding helps** — and giving you the tool to do it.

---

## ⚡ Quick Start

```bash
pip install respawn
```

```python
from respawn import recover, point_attributor

# Tell respawn where you think the failure originated
attributor = point_attributor(step=4, confidence=0.6)

# Tell respawn how to re-run from any step
def reexecute(from_step):
    new = your_engine.resume_from(from_step, retry_differently=True)
    return new.succeeded, new

# Let respawn find and fix it
result = recover(trajectory, attributor, reexecute, budget=20)
print(result.explain())
```

### Protect any LLM call from transient failures

```python
from respawn import respawn_chat

# Automatically handles: rate limits, timeouts, bad output, auth errors
result = respawn_chat(
    client,
    messages=[...],
    model="claude-haiku-4-5",
    escalate=["claude-haiku-4-5", "claude-sonnet-4-6"],  # upgrade model if needed
    validate=my_validator
)
```

---

## 🗂️ See It In Action

Two offline demos — **no API key needed:**

### 1. `examples/csv_insights/` — *What to change*
A pipeline silently misparses a sales CSV and reports **$2,056** instead of the real **$16,117**.

- Normal retry-at-crash: ❌ still wrong
- respawn re-enters at the parse step: ✅ recovers the correct figure

```bash
python examples/csv_insights/demo.py
```

### 2. `examples/ambiguous_pipeline/` — *Where to rewind*
A 5-step pipeline where a bug propagates so that multiple steps *look* guilty. Even with an imperfect attributor (~43% accuracy), respawn recovers **58%** vs blind re-entry's **38%** (p < 0.001).

```bash
python examples/ambiguous_pipeline/run.py --trials 400
```

---

## 🔍 How respawn Fits In

| Tool | Retries… | The Gap |
|---|---|---|
| Durable execution (Temporal, DBOS) | The crashing step | Replays the same failure identically |
| Retry libraries (tenacity, …) | The crashing step, differently | Still the **wrong step** |
| Attribution research (Who&When) | Nothing — just labels the cause | Stops at the label |
| **respawn** | **The causal step, re-entered & retried** | ✅ Fixes it when in-place reasoning can't |

`respawn.recover()` samples re-entry points from the attributor's probability distribution, with a uniform-exploration fallback so a wrong attributor can't permanently trap the search. Full method: [`docs/method.md`](docs/method.md)

---

## 🧪 Reproduce Everything

```bash
# The measurement — no API key needed
python experiments/whoandwhen.py --data path/to/Who\&When

# Analytic model + parameter sweeps
python experiments/run_recoverybench.py

# Reasoning-trace test (result: null — rewinding doesn't help here)
python experiments/run_probe.py --provider anthropic --model claude-haiku-4-5

# Pipeline fault test (result: significant — 93% vs 79%)
python experiments/run_recoverylab.py --provider anthropic --model claude-haiku-4-5 --n 150
```

Every experiment prints a McNemar significance verdict. No black boxes.

---

## 🗺️ Roadmap

- [x] Who&When measurement of upstream-cause prevalence
- [x] Live re-execution probe on reasoning traces (null result — honest)
- [x] Controlled pipeline test showing significant improvement
- [x] Real runnable examples for both halves of the claim
- [ ] AgenTracer + LLM-judge adapters behind the `attributor` interface
- [ ] Durable-execution adapters for Temporal / DBOS
- [ ] Side-effect compensation for non-re-enterable steps

---

## 🤝 Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📎 Citation

If you use this in your research:

```
See CITATION.cff
```

Also cite the foundational work:
- Zhang et al., *Which Agent Causes Task Failures and When?* (Who&When), ICML 2025
- *AgenTracer*, 2025

---

## License

MIT — see [LICENSE](LICENSE)
