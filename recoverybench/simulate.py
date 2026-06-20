"""Scenario generation and batch simulation."""
from __future__ import annotations

import random
from typing import Callable

from recoverybench.model import FailureEvent, RecoveryResult, Scenario, Step

ARCHETYPES = ["planner", "retriever", "executor", "critic", "orchestrator"]
FAILURE_KINDS = ["transient", "semantic", "tool", "policy"]


def _make_trace(archetype: str, n_steps: int, rng: random.Random) -> list[Step]:
    steps = []
    for i in range(n_steps):
        role = "agent" if i % 2 == 0 else "tool"
        steps.append(
            Step(
                index=i,
                role=role,
                content=f"{archetype} step {i}",
                cost_tokens=rng.randint(50, 300),
                timestamp=float(i),
            )
        )
    return steps


def generate_scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    archetype = rng.choice(ARCHETYPES)
    n_steps = rng.randint(8, 20)
    trace = _make_trace(archetype, n_steps, rng)
    fail_step = rng.randint(n_steps // 2, n_steps - 1)
    kind = rng.choice(FAILURE_KINDS)

    # Ground truth: semantic failures cause step is upstream; others are at fail_step
    if kind == "semantic":
        gt_reentry = max(0, fail_step - rng.randint(2, 5))
        gt_strategy = "truncate_and_replay"
    elif kind == "transient":
        gt_reentry = fail_step
        gt_strategy = "patch_and_continue"
    elif kind == "tool":
        gt_reentry = fail_step
        gt_strategy = "truncate_and_replay"
    else:  # policy
        gt_reentry = fail_step
        gt_strategy = "escalate"

    failure = FailureEvent(kind=kind, step_index=fail_step, message=f"Simulated {kind} error")
    return Scenario(
        id=f"scenario_{seed:04d}",
        archetype=archetype,
        trace=trace,
        failure=failure,
        ground_truth_reentry=gt_reentry,
        ground_truth_strategy=gt_strategy,
    )


def run_scenario(
    scenario: Scenario,
    controller: Callable[[Scenario], RecoveryResult],
) -> RecoveryResult:
    return controller(scenario)


def run_benchmark(
    scenarios: list[Scenario],
    controllers: dict[str, Callable[[Scenario], RecoveryResult]],
) -> dict[str, list[RecoveryResult]]:
    results: dict[str, list[RecoveryResult]] = {name: [] for name in controllers}
    for scenario in scenarios:
        for name, ctrl in controllers.items():
            results[name].append(run_scenario(scenario, ctrl))
    return results
