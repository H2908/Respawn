from __future__ import annotations

from respawn.failures import FailureKind
from respawn.strategies import Strategy


def select_strategy(kind: FailureKind, reentry_point: int, failing_step: int) -> Strategy:
    """Map (failure kind, attribution result) to a recovery strategy."""
    if kind == FailureKind.TRANSIENT:
        # Retry at the same step — no context change needed.
        return Strategy.PATCH_AND_CONTINUE

    if kind == FailureKind.POLICY:
        # Policy violations rarely benefit from replay; escalate for human review.
        return Strategy.ESCALATE

    steps_to_rewind = failing_step - reentry_point
    if kind == FailureKind.TOOL:
        # Tool errors are usually local; truncate only if cause is upstream.
        return Strategy.TRUNCATE_AND_REPLAY if steps_to_rewind > 0 else Strategy.PATCH_AND_CONTINUE

    # Semantic drift — always truncate back to the cause step.
    return Strategy.TRUNCATE_AND_REPLAY
