"""RecoveryBench data model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    index: int
    role: str               # "agent" | "tool" | "user"
    content: str
    tool_call: dict | None = None
    tool_result: dict | None = None
    cost_tokens: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
            "cost_tokens": self.cost_tokens,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Step:
        return cls(**d)


@dataclass
class FailureEvent:
    kind: str               # transient | semantic | tool | policy
    step_index: int
    message: str
    injected: bool = True


@dataclass
class Scenario:
    id: str
    archetype: str          # planner | retriever | executor | critic | orchestrator
    trace: list[Step]
    failure: FailureEvent
    ground_truth_reentry: int
    ground_truth_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def trace_as_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self.trace]


@dataclass
class RecoveryResult:
    scenario_id: str
    strategy_used: str
    reentry_point: int
    task_completed: bool
    calls_after_failure: int
    calls_baseline: int
    latency_seconds: float

    @property
    def wasted_call_ratio(self) -> float:
        if self.calls_baseline == 0:
            return 0.0
        return self.calls_after_failure / self.calls_baseline
