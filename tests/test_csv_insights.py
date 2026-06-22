"""Test the CSV insights showcase end to end (offline, deterministic)."""
import os
import sys

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "examples", "csv_insights")
sys.path.insert(0, EX)

from pipeline import build_pipeline, validate  # noqa: E402

from respawn import point_attributor, recover  # noqa: E402

CSV = os.path.join(EX, "sales.csv")
TRUE_TOTAL = 16116.88


def _localize(traj):
    for e in traj:
        if e["good_fraction"] < 0.9:
            return e["step"]
    return traj[-1]["step"]


def test_default_run_is_silently_wrong():
    pipe = build_pipeline(CSV)
    pipe.run()
    assert not validate(pipe.final)                 # quality gate catches it
    assert pipe.final["total"] < TRUE_TOTAL / 2     # badly undercounted


def test_retry_at_crash_does_not_fix_it():
    pipe = build_pipeline(CSV)
    pipe.run()
    before = pipe.final["total"]
    pipe.run_from(len(pipe.steps) - 1)              # re-run only summarize
    assert pipe.final["total"] == before            # unchanged
    assert not validate(pipe.final)


def test_respawn_recovers_correct_total():
    pipe = build_pipeline(CSV)
    traj = pipe.run()
    suspect = _localize(traj)
    assert pipe.steps[suspect].name == "parse_amount"

    def reexecute(from_step):
        step = pipe.steps[from_step]
        if step.idx + 1 >= len(step.strategies):
            return False, traj
        step.idx += 1
        new_traj = pipe.run_from(from_step)
        return validate(pipe.final), new_traj

    res = recover(traj, point_attributor(suspect, 0.9), reexecute, budget=10, explore=0.0)
    assert res.recovered
    assert abs(pipe.final["total"] - TRUE_TOTAL) < 0.01
    assert pipe.final["top_category"] == "Furniture"