# Ambiguous-attribution pipeline — testing *where* to rewind

The CSV example tests "what to change" (one obvious culprit). This one tests the
harder half of respawn's claim: **deciding *where* to rewind when the cause is
ambiguous.**

A bug at step `k` of a 5-step numeric pipeline propagates downstream, so steps
`k..4` all look anomalous — there is no single obvious culprit. A real, *imperfect*
heuristic attributor scores each step by the **new** deviation it introduced.
`recover()` can only fix the run if it re-enters at the TRUE buggy step, so
recovery rate directly measures whether the where-decision works.

Three controllers, all driven by the real `respawn.recover()`, under a budget too
small to try every step:

| controller | how it picks where to re-enter | recovery |
| --- | --- | ---: |
| retry-at-crash | only the final step | **0%** (cause is upstream) |
| blind | uniform random re-entry | ~38% |
| **respawn** | the heuristic posterior + exploration | **~58%** |

```
python examples/ambiguous_pipeline/run.py --trials 400
```

Typical output (Haiku-free, fully deterministic):

```
heuristic attributor top-1 accuracy : 43.5%   (random = 20%)
retry-at-crash recovery             :  0.0%
blind re-entry recovery             : 37.5%
respawn (where-guided) recovery     : 58.5%
gap (respawn - blind) = +21.0%   95% CI [+14.4%, +27.6%]   McNemar p<0.001   n=400
VERDICT: SIGNIFICANT — respawn beats blind.
```

The key honesty: the attributor is **wrong more than half the time** (43%
top-1), yet respawn still beats blind significantly — because `recover()`
concentrates its limited budget near the attributor's guess and falls back to
exploration when that guess is wrong. Sweep `--noise` to see the gap scale with
attribution quality: a near-random attributor (28%) still beats blind (~50% vs
38%); a good one (60%) opens the gap further (~65% vs 38%).

This reproduces RecoveryBench's simulated result on a real, executing pipeline —
closing the gap the CSV demo left: respawn doesn't just retry differently, it
**decides where to rewind**, and that decision pays off even when imperfect.