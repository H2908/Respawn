"""Truncate-and-continue re-execution.

A recovery attempt re-enters a failed trajectory at step `j`: we cut the
transcript to the first `j` messages and have a model continue the task to a
final answer. We then check that answer against ground truth.

Two continuation backends:
  * MockContinuation -- no API key. Recovers with the analytic probability
    fix_prob*(1-break)**(crash-j) for j<=root, else fails. This exists ONLY to
    test the harness end to end and to show what a real run WOULD look like if
    the analytic model held. It is NOT validation (it reproduces the model by
    construction).
  * make_real_continuation(client, model) -- the actual measurement. Uses a real
    anthropic/openai client (reusing respawn.llm's provider plumbing) to continue
    from the truncated transcript and return the model's answer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, List

from .verify import is_correct


@dataclass
class ProbeTask:
    id: str
    question: str
    ground_truth: str
    history: List[dict]          # Who&When transcript: [{content, role, name}, ...]
    root: int                    # 0-indexed decisive error step (mistake_step)
    system_prompt: str = ""

    @property
    def crash(self) -> int:
        return len(self.history) - 1   # failure surfaces at the terminal step

    @property
    def distance(self) -> int:
        return self.crash - self.root


# A continuation backend: (task, j) -> model's answer text
Continuation = Callable[[ProbeTask, int], str]
WRONG = "[[no-recovery]]"


def _transcript(task: ProbeTask, j: int, max_chars: int = 1200) -> str:
    """Serialize the trajectory up to (and including) re-entry step j."""
    lines = []
    for i, msg in enumerate(task.history[: j + 1]):
        who = msg.get("name") or msg.get("role") or "agent"
        content = str(msg.get("content", ""))
        if len(content) > max_chars:
            content = content[:max_chars] + " ...[truncated]"
        lines.append(f"[{i}] {who}: {content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Mock backend (offline plumbing; reproduces the analytic model by design)     #
# --------------------------------------------------------------------------- #
def make_mock_continuation(fix_prob: float = 0.5, reliability: float = 0.85,
                           seed: int = 0) -> Continuation:
    rng = random.Random(seed)
    break_prob = 1.0 - reliability

    def cont(task: ProbeTask, j: int) -> str:
        if j > task.root:
            return WRONG                       # cause untouched
        p = fix_prob * (1.0 - break_prob) ** (task.crash - j)          # (1-break)^d
        return task.ground_truth if rng.random() < p else WRONG

    return cont


# --------------------------------------------------------------------------- #
# Real backend (the actual measurement; needs an API key)                      #
# --------------------------------------------------------------------------- #
def make_real_continuation(client: Any, model: str, max_chars: int = 1200) -> Continuation:
    from respawn.llm import _call, _extract_text, _provider  # reuse plumbing
    provider = _provider(client)

    def cont(task: ProbeTask, j: int) -> str:
        prompt = (
            f"{task.system_prompt}\n\n"
            f"TASK:\n{task.question}\n\n"
            f"Partial progress so far (the run may contain a mistake — re-evaluate it):\n"
            f"{_transcript(task, j, max_chars)}\n\n"
            "Continue from here, correct any error you find, and solve the task. "
            "Respond with a single line:\nFINAL ANSWER: <answer>"
        )
        resp = _call(client, provider, model, [{"role": "user", "content": prompt}])
        return _extract_text(provider, resp)

    return cont


# --------------------------------------------------------------------------- #
# One re-entry attempt                                                          #
# --------------------------------------------------------------------------- #
def attempt_recover(task: ProbeTask, j: int, cont: Continuation) -> bool:
    """Re-enter at step j, continue, and check the answer against ground truth."""
    answer = cont(task, j)
    if answer == WRONG:
        return False
    return is_correct(answer, task.ground_truth)