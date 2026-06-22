"""Attribution-guided re-entry -- the recovery controller RecoveryBench validates.

This is the part that makes respawn different from durable execution and retry
libraries. They re-run the step that *crashed*. respawn asks an attributor where
the failure was *caused*, rolls back to there, retries that step differently,
and re-executes forward -- under a compute budget.

respawn owns the *policy* (where to re-enter, how to spend the budget). YOU own
the mechanics of rollback + replay, because those depend on your stack -- a
durable-execution engine (Temporal/DBOS), a checkpointer, or a pure-function
agent. You pass two callables:

    attributor(trajectory) -> posterior over step indices (the likely cause).
                              Plug in AgenTracer, an LLM-judge, or a heuristic.
    reexecute(from_step)    -> (recovered: bool, new_trajectory). Rolls the run
                              back to `from_step`, retries it differently, and
                              runs forward. Returns whether the run now succeeds.

    result = recover(trajectory, attributor, reexecute, budget=20)

The policy samples re-entry points from the attributor's posterior (so a good
prior concentrates budget near the true cause) and falls back to uniform when
the prior is flat -- it can never do worse than blind search on bad attribution,
which is exactly the property RecoveryBench checks.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# An attributor returns either a single most-likely step index, or a posterior
# as {step_index: weight} or a list of weights aligned to the trajectory.
Posterior = Union[int, Dict[int, float], Sequence[float]]
Attributor = Callable[[Sequence[Any]], Posterior]
# reexecute(from_step) -> (recovered, new_trajectory)
Reexecute = Callable[[int], Tuple[bool, Sequence[Any]]]


@dataclass
class RecoveryReentry:
    step: int
    cost: int
    recovered: bool


@dataclass
class RecoveryResult:
    recovered: bool
    spent: int
    reentries: List[RecoveryReentry] = field(default_factory=list)
    trajectory: Optional[Sequence[Any]] = None

    def explain(self) -> str:
        head = (f"respawn: {'RECOVERED' if self.recovered else 'gave up'} "
                f"after re-entering {len(self.reentries)} time(s), "
                f"spent {self.spent} step-execution(s)")
        lines = [head]
        for r in self.reentries:
            mark = "recovered" if r.recovered else "failed"
            lines.append(f"  re-enter @ step {r.step} (cost {r.cost}) -> {mark}")
        return "\n".join(lines)


def _normalize(posterior: Posterior, n: int) -> List[float]:
    """Turn any attributor output into a length-n probability vector over steps."""
    if isinstance(posterior, int):
        w = [0.0] * n
        w[max(0, min(n - 1, posterior))] = 1.0
        return w
    if isinstance(posterior, dict):
        w = [0.0] * n
        for k, v in posterior.items():
            if 0 <= k < n:
                w[k] = max(0.0, float(v))
    else:
        w = [max(0.0, float(x)) for x in posterior][:n]
        w += [0.0] * (n - len(w))
    s = sum(w)
    return [x / s for x in w] if s > 0 else [1.0 / n] * n


def recover(
    trajectory: Sequence[Any],
    attributor: Attributor,
    reexecute: Reexecute,
    budget: int,
    crash_step: Optional[int] = None,
    rng: Optional[random.Random] = None,
    explore: float = 0.15,
) -> RecoveryResult:
    """Recover a failed run by attribution-guided re-entry under a step budget.

    Args:
        trajectory:  the executed steps (any per-step records you can roll back to).
        attributor:  maps the trajectory to a posterior over the causal step.
        reexecute:   rolls back to a step, retries differently, runs forward;
                     returns (recovered, new_trajectory).
        budget:      max step-executions to spend (re-entering at step j costs
                     crash_step - j + 1).
        crash_step:  index where the failure surfaced (default: last step).
        explore:     probability of an uninformed uniform re-entry each attempt,
                     so a confidently-wrong attributor can't trap the search.

    Returns: RecoveryResult with the outcome and a re-entry receipt.
    """
    rng = rng or random.Random()
    n = len(trajectory)
    if n == 0:
        return RecoveryResult(False, 0)
    crash = (n - 1) if crash_step is None else crash_step
    post = _normalize(attributor(trajectory), n)
    steps = list(range(n))

    spent = 0
    reentries: List[RecoveryReentry] = []
    while True:
        if rng.random() < explore or sum(post) == 0:
            j = rng.randint(0, crash)                      # uninformed fallback
        else:
            j = rng.choices(steps, weights=post, k=1)[0]   # sample from the prior
            j = min(j, crash)
        cost = crash - j + 1
        if spent + cost > budget:
            break
        spent += cost
        recovered, new_traj = reexecute(j)
        reentries.append(RecoveryReentry(step=j, cost=cost, recovered=recovered))
        if recovered:
            return RecoveryResult(True, spent, reentries, new_traj)
    return RecoveryResult(False, spent, reentries, trajectory)


# --- convenience attributors ------------------------------------------------
def uniform_attributor(trajectory: Sequence[Any]) -> List[float]:
    """No information: every step equally likely. recover() reduces to blind search."""
    n = len(trajectory)
    return [1.0 / n] * n if n else []


def point_attributor(step: int, confidence: float = 0.8) -> Attributor:
    """A guess at `step` with `confidence`; remaining mass spread uniformly."""
    def attr(trajectory: Sequence[Any]) -> List[float]:
        n = len(trajectory)
        if n == 0:
            return []
        base = (1.0 - confidence) / n
        w = [base] * n
        w[max(0, min(n - 1, step))] += confidence
        return w
    return attr