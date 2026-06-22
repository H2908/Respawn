"""Recovery controllers. Each spends a step-execution budget trying to recover a
failed Task, choosing WHERE to re-enter the trajectory on each attempt.

A controller is a function: (task, budget, rng) -> (recovered: bool, spent: int).
All share the same per-attempt success model from Task; they differ ONLY in the
policy for choosing the re-entry point j. That is the whole experiment.
"""
from __future__ import annotations

import random
from typing import Callable, Tuple

from .model import Task

Controller = Callable[[Task, int, random.Random], Tuple[bool, int]]


def _attempt(task: Task, j: int, rng: random.Random) -> bool:
    return rng.random() < task.attempt_success_prob(j)


def retry_at_crash(task: Task, budget: int, rng: random.Random) -> Tuple[bool, int]:
    """Durable-execution / retry-library baseline: only ever re-run the crash step."""
    spent = 0
    while spent + task.attempt_cost(task.crash) <= budget:
        spent += task.attempt_cost(task.crash)
        if _attempt(task, task.crash, rng):
            return True, spent
    return False, spent


def rollback_all(task: Task, budget: int, rng: random.Random) -> Tuple[bool, int]:
    """Blind full restart: always re-enter at step 1 (re-run the whole prefix)."""
    spent = 0
    while spent + task.attempt_cost(1) <= budget:
        spent += task.attempt_cost(1)
        if _attempt(task, 1, rng):
            return True, spent
    return False, spent


def blind_search(task: Task, budget: int, rng: random.Random) -> Tuple[bool, int]:
    """Uninformed re-entry: pick j uniformly among candidate steps each attempt."""
    spent = 0
    while True:
        j = rng.randint(1, task.crash)
        if spent + task.attempt_cost(j) > budget:
            break
        spent += task.attempt_cost(j)
        if _attempt(task, j, rng):
            return True, spent
    return False, spent


def attribution_guided(attr_accuracy: float) -> Controller:
    """respawn's controller: re-enter using an attribution PRIOR over the root.

    attr_accuracy in [1/crash, 1] is mixed by construction so that:
      * attr_accuracy == 1.0     -> always re-enter at the true root  (= oracle)
      * attr_accuracy == 1/crash -> re-enter uniformly                (= blind_search)
    so the controller can NEVER beat blind search when attribution carries no
    information. Between those, with weight w it re-enters at the true root,
    otherwise uniformly -- modelling a noisy attributor that is right a w-fraction
    of the time. The claim under test is that useful recovery appears well below
    attr_accuracy = 1.
    """
    def controller(task: Task, budget: int, rng: random.Random) -> Tuple[bool, int]:
        floor = 1.0 / task.crash
        w = 0.0 if task.crash <= 1 else max(
            0.0, (attr_accuracy - floor) / (1.0 - floor))
        spent = 0
        while True:
            if rng.random() < w:
                j = task.root                      # attributor points at the cause
            else:
                j = rng.randint(1, task.crash)     # fall back to uninformed
            if spent + task.attempt_cost(j) > budget:
                break
            spent += task.attempt_cost(j)
            if _attempt(task, j, rng):
                return True, spent
        return False, spent
    return controller


def oracle(task: Task, budget: int, rng: random.Random) -> Tuple[bool, int]:
    """Upper bound: perfect attribution, always re-enter exactly at the root."""
    spent = 0
    while spent + task.attempt_cost(task.root) <= budget:
        spent += task.attempt_cost(task.root)
        if _attempt(task, task.root, rng):
            return True, spent
    return False, spent