"""Controlled data-pipeline tasks with an injectable upstream fault.

The regime where rollback is *necessary*, not just nicer: an early transform
silently corrupts an intermediate result, so re-reading the transcript and
re-answering cannot recover -- you must redo the bad transform and recompute
forward. This is the experiment respawn's thesis actually lives or dies on.

A task is a tiny table + a multi-hop question with a known correct answer. The
agent's solution is a PLAN: an ordered list of operations executed over the rows.
We inject a fault by corrupting one early operation's argument to a wrong-but-
valid value; downstream ops compute on the corrupted subset; the final answer is
wrong but plausible (still a number), and the fault is UPSTREAM of the answer.

Operations (a tiny, safe DSL -- no arbitrary code execution):
    {"op": "filter", "column": <col>, "value": <v>}   keep rows where col == v
    {"op": "revenue"}                                  add revenue = units*price
    {"op": "sum", "column": <col>}                     terminal: sum a column
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Row = Dict[str, Any]
Plan = List[Dict[str, Any]]


@dataclass
class Task:
    table: List[Row]
    product: str
    region: str
    question: str
    answer: float                 # ground truth
    plan: Plan                    # the correct plan
    fault_step: int               # index in plan that we corrupt
    products: List[str]
    regions: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def crash_step(self) -> int:
        return len(self.plan) - 1          # answer surfaces at the terminal op

    @property
    def distance(self) -> int:
        return self.crash_step - self.fault_step


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #
def execute(table: List[Row], plan: Plan) -> Tuple[Optional[float], List[int]]:
    """Run a plan over the table. Returns (final_scalar, rows_remaining_per_step)."""
    rows = list(table)
    trace = []
    result: Optional[float] = None
    for op in plan:
        kind = op.get("op")
        if kind == "filter":
            rows = [r for r in rows if r.get(op["column"]) == op["value"]]
        elif kind == "revenue":
            rows = [{**r, "revenue": r["units"] * r["price"]} for r in rows]
        elif kind == "sum":
            result = float(sum(r.get(op["column"], 0) for r in rows))
        trace.append(len(rows))
    return result, trace


def inject_fault(task: Task, rng: random.Random) -> Plan:
    """Corrupt the fault_step filter to a different valid value (silent + upstream)."""
    plan = [dict(op) for op in task.plan]
    op = plan[task.fault_step]
    col = op["column"]
    domain = task.products if col == "product" else task.regions
    wrong = [v for v in domain if v != op["value"]]
    op["value"] = rng.choice(wrong)
    return plan


def verify(answer: Optional[float], task: Task, eps: float = 1e-6) -> bool:
    return answer is not None and abs(answer - task.answer) < eps


# --------------------------------------------------------------------------- #
# Task generation                                                             #
# --------------------------------------------------------------------------- #
_PRODUCTS = ["Widget", "Gadget", "Gizmo", "Sprocket"]
_REGIONS = ["North", "South", "East", "West"]


def make_task(rng: random.Random, n_rows: int = 30) -> Task:
    products = list(_PRODUCTS)
    regions = list(_REGIONS)
    table = [{
        "id": i,
        "product": rng.choice(products),
        "region": rng.choice(regions),
        "units": rng.randint(1, 9),
        "price": rng.randint(2, 20),
    } for i in range(n_rows)]

    # target a product that has several matching rows, so the final sum is over
    # many rows with large numbers -- one-shot re-derivation is then error-prone,
    # while fixing one filter value + deterministic recompute stays exact.
    counts = {p: sum(1 for r in table if r["product"] == p) for p in products}
    product = max(counts, key=counts.get)
    region = ""                                # not used in the question
    plan: Plan = [
        {"op": "filter", "column": "product", "value": product},
        {"op": "revenue"},
        {"op": "sum", "column": "revenue"},
    ]
    answer, _ = execute(table, plan)
    fault_step = 0                             # the product filter is the cause
    return Task(table=table, product=product, region=region,
                question=(f"What is the total revenue (units x price) summed over "
                          f"ALL rows for product '{product}'?"),
                answer=answer, plan=plan, fault_step=fault_step,
                products=products, regions=regions)