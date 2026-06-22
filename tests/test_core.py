"""Tests for the single-step retry-differently core."""
from respawn import Escalate, FailureType, Policy, Respawn, WeakOutput


def test_reframe_recovers_bad_output():
    def step(a):
        if "Fix exactly that" not in a.feedback:
            raise WeakOutput("missing field 'total'")
        return "ok"

    r = Respawn(budget=4).run(step, model="small")
    assert r.success
    assert r.result == "ok"
    assert r.attempts == 2


def test_escalate_climbs_model_ladder():
    def step(a):
        if a.model != "large":
            raise WeakOutput(f"weak from {a.model}")
        return a.model

    pol = Policy(ladders={FailureType.BAD_OUTPUT: [Escalate(), Escalate()]})
    r = Respawn(budget=4, policy=pol).run(step, model="small")
    assert r.success
    assert r.result == "large"


def test_gives_up_when_budget_exhausted():
    def step(a):
        raise RuntimeError("always")

    r = Respawn(budget=1).run(step, model="small")
    assert not r.success
    assert r.error is not None


def test_success_first_try():
    r = Respawn().run(lambda a: 42, model="small")
    assert r.success and r.result == 42 and r.attempts == 1
    assert "OK" in r.explain()