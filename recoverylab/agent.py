"""Recovery agents for the controlled pipeline experiment.

Two conditions, given the IDENTICAL information (table + executed plan + wrong
result). The only difference is the move:

  fix_answer(task, plan, wrong)  -> float    # retry-at-crash: re-answer in place
  fix_step(task, plan, k)        -> dict      # respawn: redo the faulty step k

For respawn, the runner then re-executes the plan forward from the corrected
step deterministically. So this isolates exactly the thesis: is redoing the
single causal step + recomputing forward more reliable than re-deriving the
final answer in one shot?

Backends:
  MockAgent  -- offline plumbing. retry-at-crash fails, respawn succeeds, just to
                exercise both code paths + the verifier. NOT a result.
  RealAgent  -- real anthropic/openai calls (reuses respawn.llm plumbing).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .engine import Plan, Row, Task

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def format_table(table: List[Row]) -> str:
    cols = ["id", "product", "region", "units", "price"]
    head = " | ".join(cols)
    rows = [" | ".join(str(r[c]) for c in cols) for r in table]
    return head + "\n" + "\n".join(rows)


def format_plan(plan: Plan) -> str:
    return "\n".join(f"  step {i}: {op}" for i, op in enumerate(plan))


# --------------------------------------------------------------------------- #
# Mock backend (offline plumbing)                                             #
# --------------------------------------------------------------------------- #
class MockAgent:
    label = "MOCK (plumbing only)"

    def fix_answer(self, task: Task, plan: Plan, wrong: Optional[float]) -> Optional[float]:
        return wrong            # in-place re-answer "fails": returns the wrong value

    def fix_step(self, task: Task, plan: Plan, k: int) -> Optional[Dict[str, Any]]:
        # respawn "succeeds": returns the correct value for the faulty filter
        return {"op": "filter", "column": task.plan[k]["column"],
                "value": task.plan[k]["value"]}


# --------------------------------------------------------------------------- #
# Real backend (the genuine measurement)                                       #
# --------------------------------------------------------------------------- #
class RealAgent:
    def __init__(self, client: Any, model: str):
        from respawn.llm import _call, _extract_text, _provider
        self._call, self._text, self._provider = _call, _extract_text, _provider(client)
        self.client, self.model = client, model
        self.label = f"REAL (model={model})"

    def _ask(self, prompt: str) -> str:
        resp = self._call(self.client, self._provider, self.model,
                          [{"role": "user", "content": prompt}])
        return self._text(self._provider, resp)

    def fix_answer(self, task: Task, plan: Plan, wrong: Optional[float]) -> Optional[float]:
        prompt = (
            f"DATA TABLE:\n{format_table(task.table)}\n\n"
            f"QUESTION: {task.question}\n\n"
            f"An analysis pipeline ran these steps:\n{format_plan(plan)}\n"
            f"and produced the answer {wrong}, which is WRONG.\n\n"
            "Using the data table, compute the CORRECT answer to the question.\n"
            "Respond with ONLY:\nFINAL ANSWER: <number>"
        )
        m = _NUM.search(self._ask(prompt).split("FINAL ANSWER")[-1])
        return float(m.group()) if m else None

    def fix_step(self, task: Task, plan: Plan, k: int) -> Optional[Dict[str, Any]]:
        prompt = (
            f"DATA TABLE:\n{format_table(task.table)}\n\n"
            f"QUESTION: {task.question}\n\n"
            f"An analysis pipeline ran these steps:\n{format_plan(plan)}\n"
            f"and produced a wrong result. The operation at step {k} is wrong:\n"
            f"  {plan[k]}\n\n"
            "Give the CORRECTED value for that filter step so the pipeline answers "
            "the question. Respond with ONLY JSON:\n"
            '{"column": "<column>", "value": "<value>"}'
        )
        text = self._ask(prompt)
        try:
            blob = text[text.index("{"): text.rindex("}") + 1]
            obj = json.loads(blob)
            return {"op": "filter", "column": obj["column"], "value": obj["value"]}
        except Exception:
            return None