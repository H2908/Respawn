"""Tests for respawn.recover -- attribution-guided re-entry."""
import random

from respawn import point_attributor, recover, uniform_attributor


def make_reexecute(true_root, n):
    """Recovers iff re-entered at or before the true causal step."""
    def reexecute(j):
        return (j <= true_root, list(range(n)))
    return reexecute


def test_recovers_when_attributor_points_at_cause():
    n, root = 10, 4
    res = recover(list(range(n)), point_attributor(root, 0.9),
                  make_reexecute(root, n), budget=60,
                  rng=random.Random(0), explore=0.0)
    assert res.recovered
    assert res.reentries[-1].recovered


def test_respects_budget_and_does_not_overspend():
    n = 10  # cause at the very start -> each re-entry costs the whole trajectory
    res = recover(list(range(n)), point_attributor(0, 1.0),
                  make_reexecute(0, n), budget=5,
                  rng=random.Random(0), explore=0.0)
    assert not res.recovered
    assert res.spent <= 5


def test_uniform_attributor_runs_as_blind_search():
    n, root = 8, 3
    res = recover(list(range(n)), uniform_attributor,
                  make_reexecute(root, n), budget=80, rng=random.Random(1))
    assert isinstance(res.recovered, bool)
    assert res.spent <= 80


def test_explore_escapes_a_confidently_wrong_attributor():
    # attributor insists on step 9 (downstream of the real cause at 2);
    # exploration must still find a recovering re-entry.
    n, root = 10, 2
    res = recover(list(range(n)), point_attributor(9, 1.0),
                  make_reexecute(root, n), budget=400,
                  rng=random.Random(3), explore=0.3)
    assert res.recovered


def test_empty_trajectory_is_safe():
    res = recover([], point_attributor(0), lambda j: (True, []), budget=10)
    assert not res.recovered and res.spent == 0