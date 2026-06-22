"""A small, dependency-free CSV → insights pipeline that respawn can recover.

The pipeline computes total revenue and the top category from a messy sales CSV.
The `parse_amount` step has a silent fault: the naive strategy can't read amounts
like "$1,200.00" and zeroes them, so the downstream total is badly undercounted
WITHOUT crashing. Re-running the summary step can't fix that — the corruption is
upstream. respawn re-enters at `parse_amount`, switches to a robust strategy, and
recomputes forward.

Each step has an ordered ladder of strategies (its "retry differently" options),
and the pipeline checkpoints state after every step so it can roll back to any
step and recompute forward — exactly what respawn.recover() drives.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

State = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Strategies for each step (the "retry differently" options)                  #
# --------------------------------------------------------------------------- #
def _load(path: str) -> Callable[[State], State]:
    def step(s: State) -> State:
        with open(path, newline="", encoding="utf-8") as f:
            s["rows"] = list(csv.DictReader(f))
        return s
    return step


def parse_naive(s: State) -> State:
    for r in s["rows"]:
        try:
            r["amount_val"] = float(r["amount"])
        except ValueError:
            r["amount_val"] = 0.0            # silently drops "$1,200.00" -> 0.0
    return s


def parse_robust(s: State) -> State:
    for r in s["rows"]:
        cleaned = r["amount"].replace("$", "").replace(",", "").strip()
        try:
            r["amount_val"] = float(cleaned)
        except ValueError:
            r["amount_val"] = 0.0
    return s


def filter_complete(s: State) -> State:
    s["rows"] = [r for r in s["rows"] if r.get("status") == "complete"]
    return s


def aggregate(s: State) -> State:
    agg: Dict[str, float] = {}
    for r in s["rows"]:
        agg[r["category"]] = agg.get(r["category"], 0.0) + r.get("amount_val", 0.0)
    s["by_category"] = agg
    return s


def summarize(s: State) -> State:
    agg = s.get("by_category", {})
    s["total"] = round(sum(agg.values()), 2)
    s["top_category"] = max(agg, key=agg.get) if agg else None
    return s


# --------------------------------------------------------------------------- #
# Pipeline with checkpoints + per-step strategy ladders                       #
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    name: str
    strategies: List[Callable[[State], State]]
    idx: int = 0          # which strategy is currently selected

    @property
    def strategy_name(self) -> str:
        return self.strategies[self.idx].__name__


@dataclass
class Pipeline:
    steps: List[Step]
    checkpoints: List[State] = field(default_factory=list)   # state BEFORE each step

    def _quality(self, s: State) -> float:
        """Among rows that have been PARSED, the fraction whose non-empty amount
        became non-zero. Before the parse step nothing is parsed -> 1.0 (no
        evidence of a problem yet). A bad parse strategy collapses this."""
        rows = s.get("rows", [])
        parsed = [r for r in rows if "amount_val" in r and str(r.get("amount", "")).strip()]
        if not parsed:
            return 1.0
        ok = sum(1 for r in parsed if r["amount_val"] != 0.0)
        return ok / len(parsed)

    def run_from(self, start: int) -> List[dict]:
        """Run from step `start` using current strategies; rebuild the trajectory."""
        import copy
        cp = self.checkpoints[start] if start < len(self.checkpoints) else None
        s = copy.deepcopy(cp) if cp is not None else {}
        traj: List[dict] = []
        # keep trajectory entries for steps before `start` (unchanged)
        for i in range(start):
            traj.append(self._traj_cache[i])
        for i in range(start, len(self.steps)):
            self.checkpoints[i] = copy.deepcopy(s)
            s = self.steps[i].strategies[self.steps[i].idx](s)
            entry = {"step": i, "name": self.steps[i].name,
                     "strategy": self.steps[i].strategy_name,
                     "good_fraction": round(self._quality(s), 3)}
            traj.append(entry)
        self.final = s
        self._traj_cache = traj
        return traj

    def run(self) -> List[dict]:
        self.checkpoints = [None] * len(self.steps)
        self._traj_cache = [None] * len(self.steps)
        return self.run_from(0)


def build_pipeline(csv_path: str) -> Pipeline:
    return Pipeline(steps=[
        Step("load", [_load(csv_path)]),
        Step("parse_amount", [parse_naive, parse_robust]),   # naive first (the fault)
        Step("filter_complete", [filter_complete]),
        Step("aggregate", [aggregate]),
        Step("summarize", [summarize]),
    ])


def validate(state: State, min_quality: float = 0.9) -> bool:
    """Data-quality gate: too many unparsed amounts => the run is untrustworthy.
    Note: this checks data quality, NOT the ground-truth answer."""
    rows_seen = state.get("rows", [])
    parsed = [r for r in rows_seen if "amount_val" in r and str(r.get("amount", "")).strip()]
    if not parsed:
        return True
    ok = sum(1 for r in parsed if r["amount_val"] != 0.0)
    return (ok / len(parsed)) >= min_quality