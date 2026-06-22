"""RecoveryBench -- generative model of a failed agent trajectory.

A trajectory has T steps. It failed: a crash surfaced at step `crash`. The
*decisive* error -- the earliest step whose correction flips the outcome to
success (the literature's definition, Zhang et al. 2025) -- is at step `root`.
The distance d = crash - root is how far upstream the true cause sits from the
symptom. When d == 0 the failure is a plain transient at the crash step (the
case existing retry tools already handle).

A "recovery attempt" re-enters the trajectory at some step j and re-executes
forward through the crash. Its success probability is derived from two agent
properties, NOT hand-tuned per controller:

  fix_prob     P(a fresh/different attempt at the decisive step gets it right)
  reliability  per-step reliability (the ~0.85 from the original problem
               statement); break_prob = 1 - reliability is the chance that
               re-executing an otherwise-correct step introduces a NEW error.

Re-entering at step j re-executes (crash - j + 1) steps. To recover you must
(a) re-enter at or before the decisive step, and (b) not break any of the other
re-executed steps:

    P(recover | re-enter at j) =
        0                                          if j > root      (cause untouched)
        fix_prob * (1 - break_prob) ** (crash - j) if j <= root

Consequences that fall out of this model, none of them rigged:
  * retry-at-the-crash-step (j = crash) can only ever recover transient
    failures (d == 0); for upstream causes its recovery is exactly 0.
  * even an oracle that always re-enters at `root` decays with distance d,
    because a longer tail must be safely re-executed: fix_prob*(1-break)**d.
  * re-entering too early (j << root) is worse than j == root, because it
    needlessly re-runs correct steps that can break. So there is a real sweet
    spot at j == root that a recovery controller has to find.
"""

from __future__ import annotations
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    T:int   # trajectory length
    crash:int   # 1-based step where the failure surfaced
    root:int # 1-based decisive error step (== crash iff transient)
    fix_prob: float   # P(fresh attempt at the decisive step succeeds)
    break_prob: float # P(re-executing a correct step breaks it) = 1 - reliability

    @property 
    def distance(self)-> int:
        return self.crash-self.root
    
    @property
    def transient(self) -> bool:
        return self.root == self.crash
    

    def attempt_success_prob(self,j:int)->float:
        """P(a single recovery attempt re-entering at step j recovers the run)."""
        if j<1 or j> self.crash:
            return 0.0
        if j>self.root:
            return 0.0
        return self.fix_prob * (1.0 - self.break_prob) ** (self.crash - j)
    
    def attempt_cost(self,j:int)->int:
        """Step-executions spent re-entering at j (re-run j..crash)."""
        return self.crash - j + 1

def sample_task(
    rng: random.Random,
    T: int = 12,
    reliability: float = 0.85,
    fix_prob: float = 0.5,
    transient_fraction: float = 0.3,
    max_distance: int = 6,
) -> Task:
    """Sample a failed trajectory.

    With prob `transient_fraction` the failure is a transient at the crash step
    (d = 0). Otherwise the decisive cause is 1..max_distance steps upstream.
    """
    crash = rng.randint(max(3, T - 2), T)            # crash lands late in the run
    break_prob = 1.0 - reliability
    if rng.random() < transient_fraction:
        root = crash                                  # d = 0
    else:
        d = rng.randint(1, min(max_distance, crash - 1))
        root = crash - d
    return Task(T=T, crash=crash, root=root, fix_prob=fix_prob, break_prob=break_prob)