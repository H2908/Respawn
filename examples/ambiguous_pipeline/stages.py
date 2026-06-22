"""A multi-step pipeline where the CAUSE is genuinely ambiguous.

Unlike the CSV demo (one obvious culprit), here a bug at step k shifts the data,
and every step AFTER k also looks anomalous because the corruption propagates.
So "which step is the cause" is a real decision among several guilty-looking
candidates — the exact problem the Who&When measurement is about, made concrete
and executable.

Each step has a correct and a buggy transform. One step (chosen per trial) is
buggy; the rest are correct. The pipeline is checkpointed so respawn can roll
back to any step and recompute forward. `recover()` only fixes the run if it
re-enters at the TRUE buggy step — so recovery success is a direct test of
"decides where to rewind".
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple

Vals = List[float]


# Each transform: (correct, buggy). Correct versions are non-trivial, so a bug at
# step k propagates THROUGH the downstream transforms — every later step then also
# looks anomalous, which is what makes localizing the true cause hard.
def _scale(c): return (lambda v: [x * 1.5 for x in v], lambda v: [x * 0.5 for x in v])
def _shift(c): return (lambda v: [x + 5.0 for x in v], lambda v: [x + 20.0 for x in v])
def _clip(c):  return (lambda v: [min(x, 80.0) for x in v],
                       lambda v: [min(x, 25.0) for x in v])
def _scale2(c): return (lambda v: [x * 0.8 for x in v], lambda v: [x * 2.5 for x in v])
def _shift2(c): return (lambda v: [x - 3.0 for x in v], lambda v: [x - 15.0 for x in v])

_FACTORIES = [_scale, _shift, _clip, _scale2, _shift2]
N_TRANSFORMS = len(_FACTORIES)               # candidate fault steps: 0..4
CRASH_STEP = N_TRANSFORMS                     # final reduce step (never the cause)


@dataclass
class Pipeline:
    base_vals: Vals
    buggy_step: int                           # the injected cause
    strategy: List[int] = field(default_factory=list)   # 0=correct,1=buggy per step
    checkpoints: List[Vals] = field(default_factory=list)
    step_means: List[float] = field(default_factory=list)
    final: float = 0.0

    def __post_init__(self):
        if not self.strategy:
            self.strategy = [1 if i == self.buggy_step else 0 for i in range(N_TRANSFORMS)]

    def _transform(self, i: int, v: Vals) -> Vals:
        correct, buggy = _FACTORIES[i](None)
        return (buggy if self.strategy[i] == 1 else correct)(v)

    def run_from(self, start: int) -> "Pipeline":
        v = copy.deepcopy(self.checkpoints[start]) if start < len(self.checkpoints) \
            else list(self.base_vals)
        for i in range(start, N_TRANSFORMS):
            if i >= len(self.checkpoints):
                self.checkpoints.append(None)
            self.checkpoints[i] = copy.deepcopy(v)
            v = self._transform(i, v)
            mean = sum(v) / len(v)
            if i < len(self.step_means):
                self.step_means[i] = mean
            else:
                self.step_means.append(mean)
        self.final = round(sum(v), 3)         # the "reduce" / answer step
        return self

    def run(self) -> "Pipeline":
        self.checkpoints = []
        self.step_means = []
        return self.run_from(0)


def reference_means(base_vals: Vals) -> Tuple[List[float], float]:
    """All-correct run: the expected per-step mean profile and the true answer."""
    p = Pipeline(base_vals, buggy_step=-1, strategy=[0] * N_TRANSFORMS).run()
    return list(p.step_means), p.final


def make_trial(rng, n_vals: int = 40) -> Pipeline:
    base = [rng.uniform(5, 40) for _ in range(n_vals)]
    buggy = rng.randrange(N_TRANSFORMS)
    return Pipeline(base, buggy_step=buggy).run()


def validate(pipe: Pipeline, true_answer: float, tol: float = 0.01) -> bool:
    return abs(pipe.final - true_answer) <= tol * max(1.0, abs(true_answer))