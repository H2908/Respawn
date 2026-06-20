from respawn.failures import FailureEvent, FailureKind, classify
from respawn.recover import _heuristic_cause, make_patch, plan
from respawn.strategies import Strategy

TRACE = [
    {"index": i, "role": "agent" if i % 2 == 0 else "tool", "content": f"step {i}",
     "tool_result": {"ok": True} if i % 2 == 1 else None}
    for i in range(10)
]


def test_classify_transient():
    exc = TimeoutError("Connection timeout")
    event = classify(exc, step_index=5)
    assert event.kind == FailureKind.TRANSIENT
    assert event.step_index == 5


def test_classify_policy():
    exc = ValueError("Content policy violation triggered")
    event = classify(exc, step_index=3)
    assert event.kind == FailureKind.POLICY


def test_heuristic_transient_returns_failing_step():
    event = FailureEvent(kind=FailureKind.TRANSIENT, step_index=7, message="timeout")
    assert _heuristic_cause(event, TRACE) == 7


def test_heuristic_semantic_walks_back():
    event = FailureEvent(kind=FailureKind.SEMANTIC, step_index=8, message="drift")
    cause = _heuristic_cause(event, TRACE)
    assert cause < 8


def test_plan_transient_no_truncation():
    event = FailureEvent(kind=FailureKind.TRANSIENT, step_index=9, message="timeout")
    recovery = plan(event, TRACE)
    assert recovery.strategy == Strategy.PATCH_AND_CONTINUE
    assert recovery.reentry_point == 9


def test_plan_semantic_truncates():
    event = FailureEvent(kind=FailureKind.SEMANTIC, step_index=8, message="drift")
    recovery = plan(event, TRACE)
    assert recovery.strategy == Strategy.TRUNCATE_AND_REPLAY
    assert recovery.reentry_point < 8


def test_make_patch_transient_has_hint():
    event = FailureEvent(kind=FailureKind.TRANSIENT, step_index=5, message="rate limit")
    patch = make_patch(event, cause_step=5, trace=TRACE)
    assert "retry_hint" in patch


def test_make_patch_semantic_has_warning():
    event = FailureEvent(kind=FailureKind.SEMANTIC, step_index=7, message="off topic")
    patch = make_patch(event, cause_step=3, trace=TRACE)
    assert "drift_warning" in patch
