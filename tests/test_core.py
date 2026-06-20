import pytest

from respawn.core import Respawn
from respawn.failures import FailureEvent, FailureKind
from respawn.strategies import Strategy

TRACE = [
    {"index": i, "role": "agent" if i % 2 == 0 else "tool", "content": f"step {i}"}
    for i in range(8)
]


def test_recover_from_exception():
    rs = Respawn()
    plan = rs.recover(TimeoutError("timeout"), trace=TRACE)
    assert plan.strategy == Strategy.PATCH_AND_CONTINUE


def test_recover_from_failure_event():
    rs = Respawn()
    event = FailureEvent(kind=FailureKind.SEMANTIC, step_index=6, message="drift")
    plan = rs.recover(failure=event, trace=TRACE)
    assert plan.strategy == Strategy.TRUNCATE_AND_REPLAY
    assert plan.reentry_point <= 6


def test_guarded_attaches_failure():
    rs = Respawn()

    @rs.guarded
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError) as exc_info:
        boom()

    assert hasattr(exc_info.value, "failure")
    assert exc_info.value.failure.kind == FailureKind.SEMANTIC


def test_policy_escalates():
    rs = Respawn()
    event = FailureEvent(kind=FailureKind.POLICY, step_index=3, message="policy violation")
    plan = rs.recover(failure=event, trace=TRACE)
    assert plan.strategy == Strategy.ESCALATE
