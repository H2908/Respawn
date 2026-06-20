"""Recovery controllers evaluated in RecoveryBench.

Each controller is a callable:
    (scenario: Scenario) -> RecoveryResult
"""
from __future__ import annotations

import time

from recoverybench.model import RecoveryResult, Scenario


def _simulate_completion(scenario: Scenario, reentry_point: int) -> tuple[bool, int]:
    """Heuristic task-completion oracle used during benchmarking.

    Returns (completed, n_calls_used).
    In real evaluation this calls an LLM; here it uses ground truth proximity.
    """
    steps_replayed = len(scenario.trace) - reentry_point
    correct_reentry = reentry_point == scenario.ground_truth_reentry
    # Simplified oracle: correct reentry → higher completion probability
    completed = correct_reentry or (steps_replayed <= 3)
    return completed, steps_replayed


def full_restart_controller(scenario: Scenario) -> RecoveryResult:
    """Baseline: always restart from step 0."""
    t0 = time.monotonic()
    calls_baseline = len(scenario.trace)
    completed, calls_used = _simulate_completion(scenario, reentry_point=0)
    return RecoveryResult(
        scenario_id=scenario.id,
        strategy_used="full_restart",
        reentry_point=0,
        task_completed=completed,
        calls_after_failure=calls_baseline,
        calls_baseline=calls_baseline,
        latency_seconds=time.monotonic() - t0,
    )


def naive_truncate_controller(scenario: Scenario) -> RecoveryResult:
    """Truncate at the failing step, replay from there."""
    t0 = time.monotonic()
    calls_baseline = len(scenario.trace)
    reentry = scenario.failure.step_index
    completed, calls_used = _simulate_completion(scenario, reentry_point=reentry)
    return RecoveryResult(
        scenario_id=scenario.id,
        strategy_used="naive_truncate",
        reentry_point=reentry,
        task_completed=completed,
        calls_after_failure=calls_used,
        calls_baseline=calls_baseline,
        latency_seconds=time.monotonic() - t0,
    )


def respawn_controller(scenario: Scenario) -> RecoveryResult:
    """Attribution-guided re-entry via Respawn."""
    from respawn.failures import FailureEvent, FailureKind
    from respawn.recover import plan

    t0 = time.monotonic()
    calls_baseline = len(scenario.trace)

    failure = FailureEvent(
        kind=FailureKind(scenario.failure.kind),
        step_index=scenario.failure.step_index,
        message=scenario.failure.message,
    )
    recovery_plan = plan(failure, scenario.trace_as_dicts())

    completed, calls_used = _simulate_completion(
        scenario, reentry_point=recovery_plan.reentry_point
    )
    return RecoveryResult(
        scenario_id=scenario.id,
        strategy_used=recovery_plan.strategy,
        reentry_point=recovery_plan.reentry_point,
        task_completed=completed,
        calls_after_failure=calls_used,
        calls_baseline=calls_baseline,
        latency_seconds=time.monotonic() - t0,
    )
