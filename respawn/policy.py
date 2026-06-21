"""Policy: which strategy to try, for which failure, on which attempt.

A Policy maps a FailureType to an *ordered ladder* of strategies. Each time the
same failure type recurs, the policy advances one rung. When the ladder for a
failure is exhausted, recovery gives up and the original error propagates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass,field
from typing import Dict,List,Optional

from .failures import FailureType
from .strategies import Backoff , Decompose , Escalate, Reframe, Strategy

def default_ladders()->Dict[FailureType,List[Strategy]]:
    return {
        FailureType.BAD_OUTPUT: [Reframe(), Escalate(), Decompose()],
        FailureType.TOOL_ERROR: [Reframe(), Backoff()],
        FailureType.RATE_LIMIT: [Backoff(), Backoff()],
        FailureType.TIMEOUT:    [Backoff(), Escalate()],
        FailureType.LOOP:       [Decompose(), Escalate()],
        FailureType.FATAL:      [],  # never retry
        FailureType.UNKNOWN:    [Reframe(), Backoff()],
    }
@dataclass
class Policy:
    ladders: Dict[FailureType, List[Strategy]] = field(default_factory=default_ladders)
    _rung: Dict[FailureType, int] = field(default_factory=lambda: defaultdict(int))

    def choose(self, failure_type: FailureType) -> Optional[Strategy]:
        ladder = self.ladders.get(failure_type, [])
        i = self._rung[failure_type]
        if i >= len(ladder):
            return None
        self._rung[failure_type] += 1
        return ladder[i]