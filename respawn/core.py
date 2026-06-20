"""Top-level Respawn interface."""
from __future__ import annotations

import functools
from typing import Any, Callable

from respawn.failures import FailureEvent, classify
from respawn.recover import RecoveryPlan, plan


class Respawn:
    """Wraps an agent loop with failure classification and recovery planning."""

    def recover(self, failure: Exception | FailureEvent, trace: list[dict]) -> RecoveryPlan:
        """Classify `failure` and return a RecoveryPlan for the given trace."""
        if isinstance(failure, Exception):
            step_index = len(trace)
            event = classify(failure, step_index)
        else:
            event = failure
        return plan(event, trace)

    def guarded(self, fn: Callable) -> Callable:
        """Decorator: catch exceptions from `fn` and attach a `.failure` attribute."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                exc.failure = classify(exc, step_index=0)  # type: ignore[attr-defined]
                raise

        return wrapper
