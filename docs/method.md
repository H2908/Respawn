# Method

## Thesis

LLM agent failures are rarely total. A tool timeout in step 12 of a 20-step trace doesn't invalidate steps 1–11. Yet the default recovery strategy — full restart — discards everything and starts over. This wastes tokens, wastes time, and often triggers the same failure again.

**Respawn's thesis**: every failure has a *cause step* — the earliest step whose output, if different, would have prevented the failure. Re-entering at the cause step (with a patched context) recovers the task without replaying clean work.

The challenge is finding the cause step automatically. That's the attribution problem.

---

## The 4-layer cake

```
┌─────────────────────────────────────────┐
│  Layer 4 · Re-entry (core.py)           │  resume execution at the right step
├─────────────────────────────────────────┤
│  Layer 3 · Strategy (strategies.py,     │  decide HOW to recover
│             policy.py)                  │  (truncate | patch | escalate | abort)
├─────────────────────────────────────────┤
│  Layer 2 · Attribution (recover.py      │  decide WHERE to rewind
│             ↔ Who&When)                 │  (which step caused the failure?)
├─────────────────────────────────────────┤
│  Layer 1 · Classification (failures.py) │  decide WHAT failed
│                                         │  (transient | semantic | tool | policy)
└─────────────────────────────────────────┘
```

Each layer is independently testable and replaceable. The policy layer (Layer 3) is the integration point: it maps (FailureKind × AttributionResult) → Strategy.

---

## Division of labour

| Component | Owns | Does NOT own |
|-----------|------|--------------|
| `failures.py` | failure taxonomy and classification heuristics | attribution, strategy |
| `recover.py` | orchestrating attribution via Who&When | the attribution algorithm itself |
| Who&When (external) | step-level causal attribution | failure classification, strategy |
| AgenTracer (external) | trace collection and serialization | any recovery logic |
| `policy.py` | strategy selection rules | execution of the strategy |
| `strategies.py` | strategy implementations | which strategy to pick |
| `core.py` | re-entry interface | everything above |

The external tools (Who&When, AgenTracer) are optional. When absent, Respawn falls back to heuristic attribution (last tool-call step before the failure).

---

## Attribution-guided re-entry (the novel core)

Standard retry loops operate at the *task* level: fail → restart task. Respawn operates at the *step* level:

1. Receive a `FailureEvent` with the failing step index `f`.
2. Ask Who&When: "given this trace and this failure, which step index `c ≤ f` is the cause?"
3. Validate `c` against the failure kind (transient failures almost always have `c = f`; semantic failures can have `c ≪ f`).
4. Construct a `RecoveryPlan(reentry_point=c, patch=..., strategy=...)`.
5. Hand the plan to `core.py`, which truncates the trace at `c-1` and resumes.

The patch is a dict of context overrides applied to the agent state before re-entry. For transient failures it's empty. For semantic drift it might include a corrected sub-goal or a forbidden tool list.

---

## Why attribution matters

On RecoveryBench, semantic failures (context drift) account for 28% of scenarios but 61% of wasted tokens under naive truncate-at-failure retry — because naive truncate re-enters at `f`, which is already corrupted. Attribution finds the true cause step `c < f`, eliminating the drift before re-entry.

The 8× WCR improvement is almost entirely explained by catching these cases correctly.

---

## Limitations

- **Single failure assumption.** Recovery plans are constructed for one failure at a time. If re-entry triggers a second failure, the outer durable-execution layer must call `recover()` again.
- **Attribution oracle error.** Who&When has ~3% error rate on RecoveryBench. Mis-attributed failures lead to either over-truncation (safe but wasteful) or under-truncation (risky — may re-trigger).
- **No learned policy.** The policy layer uses hand-written rules. A learned policy (RL or distillation from human annotators) is future work.
