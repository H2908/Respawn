"""Recovery strategies.

A strategy is a small object that mutates the *next* Attempt given the failure
that just happened. This is the "retry differently" part: each one changes one
lever (prompt, model, tool, timing, scope).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Protocol

from .failures import FailureInfo

class Strategy(Protocol):
    name:str
    def apply(self, attempt: Attempt, failure: FailureInfo) -> None: ...  # noqa: F821

@dataclass
class Reframe:
    """Inject the error back into the prompt and ask for a corrected attempt."""
    name: str = "reframe"

    def apply(self,attempt, failure) -> None:
        attempt.feedback=(
            f"Your previous attempt failed: {failure}. "
            "Fix exactly that and try again."
        )

@dataclass
class Escalate:
    """Climb a model ladder; stronger model on each invocation."""
    ladder:List[str]=field(default_factory=lambda:['small','medium','large'])
    name:str ="escalate"

    def apply(self,attempt,failure)->None:
        try:
            i=self.ladder.index(attempt.model)
        except ValueError:
            i=-1
        attempt.model=self.ladder[min(i+1,len(self.ladder)-1)]
@dataclass
class SwapTool:
    """Move to the next fallback tool for the failing capability."""
    fallbacks: dict=field(default_factory=dict)
    name:str ="swap_tool"

    def apply(self,attempt,failure)->None:
        for tool, alts in self.fallbacks.items():
            cur=attempt.tools.get(tool)
            if cur in alts:
                nxt=alts.index(cur) + 1
                if nxt < len(alts):
                    attempt.tools[tool]=alts[nxt]
            elif alts:
                attempt.tools[tool]=alts[0]


@dataclass
class BackOff:
    """Exponential backoff with jitter. The only strategy that sleeps."""
    base: float = 0.5
    cap: float = 30.0
    name: str = "backoff"

    def apply(self,attempt,failure)-> None:
        import random
        delay=min(self.cap,self.base*(2**(attempt.n-1)))
        time.sleep(delay +random.uniform(0,delay))

@dataclass
class Decompose:
    """Ask the step to break the task down. Sets a flag the step can read."""
    name: str = "decompose"

    def apply(self, attempt, failure) -> None:
        attempt.decompose = True
        attempt.feedback = (
            f"Previous attempt failed: {failure}. Break the task into smaller "
            "sub-steps and solve them one at a time."
        )


