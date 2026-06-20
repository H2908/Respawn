"""Attribution-guided re-entry — the novel core of Respawn.

Given a failure event and an execution trace, this module:
  1. Calls Who&When (if available) to attribute the failure to a cause step.
  2. Falls back to heuristic attribution when Who&When is absent.
  3. Constructs a RecoveryPlan with a reentry point, context patch, and strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from respawn.failures import FailureEvent, FailureKind
from respawn.policy import select_strategy
from respawn.strategies import Strategy


@dataclass
class RecoveryPlan:
    strategy: Strategy
    reentry_point: int
    patch: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


def _heuristic_cause(failure: FailureEvent, trace: list[dict]) -> int:
    """Fallback attribution when Who&When is unavailable.

    Transient/tool failures point to the failing step itself.
    Semantic failures walk back to find the last assistant turn before drift.
    """
    if failure.kind in (FailureKind.TRANSIENT, FailureKind.TOOL):
        return failure.step_index

    # Walk backwards looking for the last step that produced a tool result,
    # as a rough proxy for where context corruption might have started.
    for i in range(failure.step_index - 1, -1, -1):
        step = trace[i] if i < len(trace) else {}
        if step.get("role") == "tool" and step.get("tool_result"):
            return max(0, i - 1)

    return 0


def _call_whoandwhen(failure: FailureEvent, trace: list[dict]) -> int | None:
    """Attempt to import and call Who&When for causal attribution.

    Returns the attributed cause step index, or None if unavailable.
    """
    try:
        from whoandwhen import attribute  # type: ignore[import]

        result = attribute(trace=trace, failure_step=failure.step_index)
        return result.cause_step
    except ImportError:
        return None


def make_patch(failure: FailureEvent, cause_step: int, trace: list[dict]) -> dict[str, Any]:
    """Build a context patch to apply before re-entry."""
    patch: dict[str, Any] = {}

    if failure.kind == FailureKind.TRANSIENT:
        patch["retry_hint"] = f"Previous attempt failed transiently: {failure.message}"
    elif failure.kind == FailureKind.TOOL:
        patch["tool_error"] = failure.message
        patch["avoid_tool_args"] = _extract_bad_args(trace, failure.step_index)
    elif failure.kind == FailureKind.SEMANTIC:
        patch["drift_warning"] = (
            f"Context drift detected starting around step {cause_step}. "
            "Re-focus on the original goal."
        )

    return patch


def _extract_bad_args(trace: list[dict], step_index: int) -> dict:
    if step_index < len(trace):
        return trace[step_index].get("tool_call", {})
    return {}


def plan(failure: FailureEvent, trace: list[dict]) -> RecoveryPlan:
    """Entry point: produce a RecoveryPlan for a given failure and trace."""
    cause_step = _call_whoandwhen(failure, trace)
    attributed_by = "who&when"
    if cause_step is None:
        cause_step = _heuristic_cause(failure, trace)
        attributed_by = "heuristic"

    strategy = select_strategy(failure.kind, cause_step, failure.step_index)
    patch = make_patch(failure, cause_step, trace)

    rationale = (
        f"Failure '{failure.kind}' at step {failure.step_index} "
        f"attributed to step {cause_step} via {attributed_by}. "
        f"Strategy: {strategy}."
    )

    return RecoveryPlan(
        strategy=strategy,
        reentry_point=cause_step,
        patch=patch,
        rationale=rationale,
    )
