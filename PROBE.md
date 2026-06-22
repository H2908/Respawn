# Live re-execution probe

Everything else in this repo measures recovery under an **analytic** model:
`P(recover | re-enter at j) = fix_prob * (1 - break_prob)^(crash - j)` for
`j <= cause`. The probe replaces that assumption with **real LLM re-runs** on
real Who&When tasks, to answer two questions:

1. Does rewinding to the **annotated cause** and re-running recover the
   ground-truth answer more often than rewinding only to the **crash step**?
2. Do the observed recovery numbers **track the analytic curve** the rest of the
   repo assumes?

## How it works: truncate-and-continue

A recovery attempt re-enters at step `j`: cut the transcript to the first `j`
messages, have a model continue the task to a `FINAL ANSWER:`, and check it
against the dataset's `ground_truth`.

We compare two re-entry points per task, with the **identical** procedure:
- `j = crash` (the terminal step) — the retry-at-crash baseline.
- `j = cause` (the annotated `mistake_step`) — respawn's move.

## What it does and does not establish

**Honest limitation.** Continuation is single-model — an approximation of the
original multi-agent run, which cannot be faithfully resumed from an arbitrary
step without the original framework wired up. So the **absolute** recovery
numbers are procedure-dependent. But the **comparison** (cause vs crash) uses
the same procedure and differs only in the cut point, so the *relative* result —
"acting on the attribution recovers runs that retrying the failing step cannot" —
is internally valid. A stricter verifier (or LLM judge) would shift absolute
numbers, not the comparison.

## Running it

**Offline (no API key)** — plumbing only. The mock backend recovers with the
analytic probability *by construction*, so it reproduces the curve and proves
the harness works end to end. It is **not** validation:

```bash
make probe         # python experiments/run_probe.py --provider mock --n 80 --trials 6
```

**The real measurement** — needs a key in your environment
(`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`):

```bash
# after: make data   (clones Who&When)
python experiments/run_probe.py --provider anthropic --model claude-haiku-4-5 \
    --n 40 --trials 3 --data "Agents_Failure_Attribution/Who&When"
```

Output: observed recovery for retry-at-crash vs rewind-to-cause, the lift, and a
figure (`docs/images/probe_results.png`) whose title records the mode and model
so a mock run can never be mistaken for a real one. A scatter of observed
recovery vs upstream distance is overlaid on the analytic curve.

## Reading the result

- **rewind-to-cause >> retry-at-crash** confirms the central claim on real
  re-runs, not just in simulation.
- **observed points near the analytic curve** validate the model the benchmark
  uses; **systematic deviation** tells you to refit `fix_prob` / `reliability`
  (and is itself a finding worth reporting).

Either way, the probe converts the headline from "validated in simulation" to
"validated against real re-runs" — the version worth sending to the Who&When
authors.