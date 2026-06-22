# Results

This project makes one solid measurement, then honestly tests a hypothesis that
follows from it — failing in one regime and succeeding in another. All three are
reported, because the boundary between them is the actual contribution.

## 1. Measurement (solid, free, reproducible)

On the **Who&When** benchmark (Zhang et al., ICML 2025) — 184 real LLM
multi-agent failures with human/algorithm annotations of the decisive error
step — the decisive error is at the **final step in only 4.3% of cases**. In
**95.7%** the cause is *upstream* of where the failure surfaced (median 5 steps
back).

So durable execution and retry libraries — which re-run the step that
**crashed** — can structurally address **at most ~4.3%** of real agent failures.

This needs no model calls; reproduce with `python experiments/whoandwhen.py`.

## 2. Hypothesis: act on the attribution by re-entering at the cause

If the cause is upstream, rewind to it and retry differently. We tested this two
ways.

### 2a. The analytic model (RecoveryBench) — a *prediction*, not a result

A synthetic model (`fix_prob·(1−break)^d`) predicts large recovery lift from
attribution-guided re-entry, robust to imperfect attribution. Over Who&When's
real distance distribution, with the literature's measured attribution
accuracies, it *projects* ~35% recovery vs 4.3% for retry-at-crash.

**This is a model projection, not a measurement.** It motivated the live tests
below; it is not evidence on its own.

### 2b. Live test on reasoning traces (Who&When probe) — NULL

When we actually re-run Who&When tasks with a real model (truncate-and-continue,
Haiku), rewinding to the annotated cause does **not** beat re-running at the
crash step (≈1.0× at d≥1; not significant). 

**Why:** on reasoning traces, a capable model re-reading the whole transcript
*self-corrects the upstream error in place* — so there is nothing for rollback to
add. The analytic model's assumption that retry-at-crash recovers ~0% of upstream
causes does not hold for real models on reasoning tasks.

This is a genuine negative result, and it sharpens the claim.

### 2c. Live test where rollback is *necessary* (RecoveryLab) — SIGNIFICANT

We then tested the regime the probe lacked: a **stateful pipeline with a silent
upstream fault**, where re-reading cannot un-corrupt the computed state. Both
conditions see the identical table + plan; only the move differs — retry-at-crash
re-answers in one shot, respawn redoes the faulty step and recomputes forward.

| condition | recovery (n=150, Haiku) |
| --- | ---: |
| retry-at-crash (re-answer in place) | **78%** |
| respawn (redo faulty step + recompute) | **95%** |

```
gap (respawn - retry) = +17%   95% CI excludes 0   McNemar p <0.01   n=150
VERDICT: SIGNIFICANT — respawn beats retry-at-crash
```

Replication (seed 1): <fill in after running --seed 1>.

**Why it works here:** respawn re-runs actual code (exact), while re-answering
forces the model to re-derive the result by mental arithmetic (error-prone). On
state-corrupting faults, rollback is necessary, not just nicer.

## 3. What this establishes

- **Most agent failures originate upstream of their symptom** (measured, solid).
- **Re-entering at the cause helps when in-place re-reasoning *cannot* fix the
  error** — i.e. state-corrupting / side-effecting faults (measured, significant).
- **It does *not* help on reasoning traces**, where models self-correct in place
  (measured, null).

The contribution is this boundary, not a single headline number. The simulated
"large lift" did **not** survive contact with real models on reasoning tasks; it
held only in the narrower regime where rollback is structurally necessary.

## Honest limits

- RecoveryLab is a controlled synthetic pipeline; the fault is injected and
  localization is given (we tell respawn which step is faulty). It isolates the
  recovery mechanism, not end-to-end attribution.
- The probe's continuation is single-model, an approximation of multi-agent
  re-execution; its *comparison* is internally valid, its absolute numbers are not.
- Numbers are Haiku; a stronger model would shift absolutes (and likely narrow
  RecoveryLab's gap as one-shot arithmetic improves — bump `--rows` to compensate).