"""Monte Carlo harness: evaluate controllers on a shared pool of failed tasks.

The same seeded task pool is used for every controller so comparisons are paired
(each controller faces identical failures), and each controller gets its own RNG
stream so its stochastic retries don't perturb the others.
"""
from __future__ import annotations

import random
from typing import Dict, List

from .controllers import Controller
from .model import Task, sample_task


def make_pool(n: int, seed: int = 0, **task_kwargs) -> List[Task]:
    rng = random.Random(seed)
    return [sample_task(rng, **task_kwargs) for _ in range(n)]


def evaluate(controller: Controller, pool: List[Task], budget: int,
             seed: int = 1234) -> Dict[str, float]:
    rng = random.Random(seed)
    recovered = 0
    spent_total = 0
    spent_on_success = 0
    for task in pool:
        ok, spent = controller(task, budget, rng)
        spent_total += spent
        if ok:
            recovered += 1
            spent_on_success += spent
    n = len(pool)
    return {
        "recovery_rate": recovered / n,
        "avg_budget_spent": spent_total / n,
        "avg_cost_per_success": (spent_on_success / recovered) if recovered else float("nan"),
    }