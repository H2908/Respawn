"""Significance for paired recovery comparisons (stdlib only).

retry-at-crash and respawn are run on the SAME tasks, so the outcomes are
paired. The right test is McNemar's, which looks only at the discordant pairs:

    b = tasks where respawn recovered but retry-at-crash did NOT
    c = tasks where retry-at-crash recovered but respawn did NOT

Concordant pairs (both win / both lose) carry no information about the
difference, so McNemar is both more correct and more powerful than an unpaired
two-proportion test. We report an exact two-sided McNemar p-value and a 95% CI
on the difference in recovery rates.

Caveat: when trials > 1 per task, pairs from the same task are not fully
independent (clustering), which slightly understates the CI. With trials = 1
each pair is one task and this is exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.96):
    """95% CI for (respawn rate - retry rate) from paired discordant counts."""
    if n == 0:
        return 0.0, 0.0, 0.0
    diff = (b - c) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n ** 2)
    se = math.sqrt(max(var, 0.0))
    return diff, diff - z * se, diff + z * se


@dataclass
class Verdict:
    n_pairs: int
    discordant: int
    diff: float
    ci_lo: float
    ci_hi: float
    p: float
    significant: bool
    note: str
    baseline: str = "retry-at-crash"

    def line(self) -> str:
        head = (f"  gap (respawn - {self.baseline}) = {100*self.diff:+.1f}%   "
                f"95% CI [{100*self.ci_lo:+.1f}%, {100*self.ci_hi:+.1f}%]   "
                f"McNemar p={self.p:.3f}   n={self.n_pairs} (discordant={self.discordant})")
        return head + f"\n  VERDICT: {self.note}"


def assess(b: int, c: int, n_pairs: int, min_discordant: int = 10,
           baseline: str = "retry-at-crash") -> Verdict:
    diff, lo, hi = paired_diff_ci(b, c, n_pairs)
    p = mcnemar_exact_p(b, c)
    discordant = b + c
    if discordant < min_discordant:
        sig, note = False, (f"INCONCLUSIVE — only {discordant} discordant pairs; "
                            "need more tasks/trials for power.")
    elif p < 0.05 and diff > 0:
        sig, note = True, f"SIGNIFICANT — respawn beats {baseline}."
    elif p < 0.05 and diff < 0:
        sig, note = False, f"SIGNIFICANT in the OTHER direction — {baseline} wins."
    elif diff > 0:
        sig, note = False, "not significant (gap positive but CI includes 0)."
    else:
        sig, note = False, "not significant — no detectable difference (null)."
    return Verdict(n_pairs, discordant, diff, lo, hi, p, sig, note, baseline)