"""Tests for the live re-execution probe harness (no API key needed)."""
from probe import ProbeTask, attempt_recover, is_correct, normalize
from probe.continuation import WRONG


def _task(root, n=8, gt="42"):
    history = [{"content": f"step {i}", "role": "assistant"} for i in range(n)]
    return ProbeTask(id="t", question="q", ground_truth=gt, history=history, root=root)


def perfect_cont(task, j):
    """Recovers exactly when re-entered at or before the cause."""
    return task.ground_truth if j <= task.root else WRONG


def test_distance_and_crash_geometry():
    t = _task(root=3, n=8)
    assert t.crash == 7
    assert t.distance == 4


def test_rewind_to_cause_beats_crash_on_upstream():
    t = _task(root=3, n=8)
    assert attempt_recover(t, t.root, perfect_cont)       # cause -> recovers
    assert not attempt_recover(t, t.crash, perfect_cont)  # crash -> cause untouched


def test_reentry_downstream_of_cause_fails():
    t = _task(root=3, n=8)
    assert not attempt_recover(t, 5, perfect_cont)        # 5 > root=3


def test_verifier_normalization():
    assert is_correct("FINAL ANSWER: 42", "42")
    assert is_correct("The answer is Paris.", "paris")
    assert not is_correct("FINAL ANSWER: 7", "42")
    assert normalize("The  Answer!!") == "answer"