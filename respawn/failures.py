from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureKind(str, Enum):
    TRANSIENT = "transient"    # retryable glitch (timeout, rate-limit, network)
    SEMANTIC = "semantic"      # context drift / wrong sub-goal
    TOOL = "tool"              # deterministic tool error (bad args, missing resource)
    POLICY = "policy"          # guardrail / content-policy violation


@dataclass
class FailureEvent:
    kind: FailureKind
    step_index: int
    message: str
    raw: Any = field(default=None, repr=False)


_TRANSIENT_MARKERS = frozenset(
    ["timeout", "rate limit", "connection", "503", "429", "temporarily"]
)
_POLICY_MARKERS = frozenset(
    ["policy", "content filter", "refusal", "not allowed", "cannot assist"]
)


def classify(exc: Exception, step_index: int) -> FailureEvent:
    msg = str(exc).lower()
    if any(m in msg for m in _TRANSIENT_MARKERS):
        kind = FailureKind.TRANSIENT
    elif any(m in msg for m in _POLICY_MARKERS):
        kind = FailureKind.POLICY
    elif "tool" in msg or "function" in msg or "argument" in msg:
        kind = FailureKind.TOOL
    else:
        kind = FailureKind.SEMANTIC
    return FailureEvent(kind=kind, step_index=step_index, message=str(exc), raw=exc)
