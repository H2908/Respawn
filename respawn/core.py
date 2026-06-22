"""Core run loop: Attempt, Receipt, and Respawn.run().

    from respawn import Respawn

    def step(a):           # your work; reads from the mutable Attempt
        ...                # raise on failure, return on success

    result = Respawn(budget=4).run(step, prompt="...", model="small")

The step receives an `Attempt` it can read (prompt, model, tools, feedback,
decompose). On failure respawn classifies why, mutates the Attempt via the
chosen strategy, and runs the step again -- differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .failures import Classifier, FailureInfo, default_classifier
from .policy import Policy


@dataclass
class Attempt:
    n: int                              # 1-based attempt number
    prompt: str = ""
    model: str = "small"
    tools: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""                  # guidance injected after a failure
    decompose: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    n: int
    failure: Optional[str]              # str(FailureInfo) or None on success
    strategy: Optional[str]             # strategy applied for the NEXT attempt


@dataclass
class Receipt:
    success: bool
    result: Any
    attempts: int
    traces: List[Trace] = field(default_factory=list)
    error: Optional[BaseException] = None

    def explain(self) -> str:
        lines = [f"respawn: {'OK' if self.success else 'FAILED'} "
                 f"after {self.attempts} attempt(s)"]
        for t in self.traces:
            if t.failure is None:
                lines.append(f"  #{t.n} -> success")
            else:
                via = f" -> retry via [{t.strategy}]" if t.strategy else " -> give up"
                lines.append(f"  #{t.n} {t.failure}{via}")
        return "\n".join(lines)


class Respawn:
    def __init__(
        self,
        budget: int = 4,
        policy: Optional[Policy] = None,
        classifier: Classifier = default_classifier,
        on_event: Optional[Callable[[str], None]] = None,
    ):
        self.budget = budget
        self.policy = policy or Policy()
        self.classify = classifier
        self.on_event = on_event or (lambda _: None)

    def run(self, step: Callable[[Attempt], Any], **initial) -> Receipt:
        attempt = Attempt(n=1, **initial)
        traces: List[Trace] = []
        last_error: Optional[BaseException] = None

        for n in range(1, self.budget + 1):
            attempt.n = n
            try:
                result = step(attempt)
                traces.append(Trace(n=n, failure=None, strategy=None))
                self.on_event(f"#{n} success")
                return Receipt(True, result, n, traces)
            except BaseException as e:  # noqa: BLE001 -- recovery is the point
                last_error = e
                failure: FailureInfo = self.classify(e, attempt)
                strategy = self.policy.choose(failure.type) if n < self.budget else None
                traces.append(Trace(n=n, failure=str(failure),
                                    strategy=strategy.name if strategy else None))
                self.on_event(f"#{n} {failure} "
                              f"-> {strategy.name if strategy else 'give up'}")
                if strategy is None:
                    break
                strategy.apply(attempt, failure)

        return Receipt(False, None, len(traces), traces, error=last_error)