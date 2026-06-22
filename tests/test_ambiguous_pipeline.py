"""Test that respawn's WHERE-to-rewind decision beats blind under ambiguity."""
import os
import random
import sys

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "examples", "ambiguous_pipeline")
sys.path.insert(0, EX)

from run import run_experiment  # noqa: E402
from stages import (  # noqa: E402
    N_TRANSFORMS,
    Pipeline,
    make_trial,
    reference_means,
    validate,
)


def test_fault_propagates_then_fixing_cause_recovers():
    rng = random.Random(0)
    t = make_trial(rng)
    _, true_ans = reference_means(t.base_vals)
    assert not validate(t, true_ans)                 # broken
    t.strategy[t.buggy_step] = 0                      # fix the true cause
    t.run_from(t.buggy_step)
    assert validate(t, true_ans)                     # recovered


def test_fixing_the_wrong_step_does_not_recover():
    rng = random.Random(1)
    t = make_trial(rng)
    _, true_ans = reference_means(t.base_vals)
    wrong = (t.buggy_step + 1) % N_TRANSFORMS
    p = Pipeline(t.base_vals, buggy_step=t.buggy_step).run()
    p.strategy[wrong] = 0                            # "fix" a non-cause step
    p.run_from(wrong)
    assert not validate(p, true_ans)                 # still broken (cause untouched)


def test_respawn_beats_blind_under_imperfect_attribution():
    r = run_experiment(trials=160, seed=0)
    # attributor is imperfect but informative (above random 20%, below oracle)
    assert 0.25 < r["attr_acc"] < 0.95
    # retry-at-crash can never fix an upstream cause
    assert r["retry"] == 0.0
    # deciding WHERE (even imperfectly) beats blind re-entry
    assert r["respawn"] > r["blind"] + 0.08