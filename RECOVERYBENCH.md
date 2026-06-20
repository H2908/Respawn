# RecoveryBench

A benchmark for evaluating LLM-agent recovery strategies under realistic failure conditions.

---

## Model

RecoveryBench generates synthetic execution traces for five **agent archetypes**:

| Archetype | Description | Typical failure mode |
|-----------|-------------|----------------------|
| `planner` | Multi-step goal decomposition | Semantic drift after tool error |
| `retriever` | RAG / search loop | Irrelevant context accumulation |
| `executor` | Sequential tool-call chain | Transient API timeout |
| `critic` | Self-review loop | Policy violation on re-attempt |
| `orchestrator` | Sub-agent coordinator | Deadlock / missing sub-agent result |

Each trace is a list of `Step` objects:

```python
@dataclass
class Step:
    index: int
    role: str           # "agent" | "tool" | "user"
    content: str
    tool_call: dict | None
    tool_result: dict | None
    cost_tokens: int
    timestamp: float
```

A **scenario** bundles a trace with an injected failure:

```python
@dataclass
class Scenario:
    id: str
    archetype: str
    trace: list[Step]
    failure: FailureEvent
    ground_truth_reentry: int   # step index a human annotator would rewind to
    ground_truth_strategy: str
```

---

## Assumptions

1. **Checkpoints are free.** We assume the durable-execution layer can snapshot/restore at any step index at negligible cost. RecoveryBench measures LLM-call savings, not checkpoint I/O.
2. **Attribution oracle.** Who&When is used as a black-box oracle. Its own error rate (~3%) is baked into the overall numbers but not separately ablated.
3. **Single-failure scenarios.** Each scenario contains exactly one injected failure. Cascading failures are future work.
4. **No human in the loop.** Escalation strategies are scored as "failed" for task completion but credited for avoiding wasted tokens.
5. **Token cost = input + output tokens at gpt-4o pricing** as of benchmark freeze (2026-Q1). Relative numbers are model-agnostic.

---

## Metrics

| Metric | Definition |
|--------|-----------|
| **TCR** | Task Completion Rate — did the agent finish the goal after recovery? |
| **WCR** | Wasted-Call Ratio — (calls after failure) / (calls in full restart baseline) |
| **LAT** | Latency overhead — wall-clock seconds from failure detection to re-entry |
| **PPT** | Patch Precision @ Top-1 — did the reentry point match ground truth? |

---

## Full results

### Overall (n = 500 scenarios)

| Strategy | TCR | WCR | LAT (s) |
|----------|-----|-----|---------|
| Full restart (baseline) | 71.2% | 1.00× | 0.0 |
| Naive truncate | 73.8% | 0.61× | 0.4 |
| **Respawn (ours)** | **75.5%** | **0.12×** | 1.1 |

Respawn improves TCR by **+4.3 pp** and reduces wasted calls by **8.3×** vs. the baseline.

### By archetype

| Archetype | Baseline TCR | Respawn TCR | WCR |
|-----------|-------------|-------------|-----|
| planner | 68.0% | 74.0% | 0.09× |
| retriever | 75.0% | 78.0% | 0.11× |
| executor | 80.0% | 82.0% | 0.08× |
| critic | 62.0% | 67.0% | 0.18× |
| orchestrator | 71.0% | 76.0% | 0.14× |

### By failure kind

| Failure kind | Baseline TCR | Respawn TCR |
|--------------|-------------|-------------|
| transient | 85.0% | 86.0% |
| semantic | 60.0% | 68.0% |
| tool | 78.0% | 80.0% |
| policy | 52.0% | 60.0% |

Semantic and policy failures benefit most — they require non-trivial reentry point selection.

---

## Reproducing

```bash
pip install "respawn[bench]"
make data        # fetch / generate the 500 scenarios into data/
make reproduce   # runs experiments/run_recoverybench.py, writes docs/images/
```

Results are written to `results/recoverybench_YYYYMMDD.json` and figures to `docs/images/`.
