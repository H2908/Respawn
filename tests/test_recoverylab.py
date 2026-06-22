"""Tests for recoverylab -- the controlled pipeline experiment."""
import random

from recoverylab import MockAgent, execute, inject_fault, make_task, verify


def test_clean_plan_gives_correct_answer():
    rng = random.Random(0)
    t = make_task(rng)
    ans, _ = execute(t.table, t.plan)
    assert verify(ans, t)


def test_injected_fault_changes_the_answer():
    rng = random.Random(1)
    changed = 0
    for _ in range(50):
        t = make_task(rng)
        bad_plan = inject_fault(t, rng)
        ans, _ = execute(t.table, bad_plan)
        if not verify(ans, t):
            changed += 1
    # the great majority of injected faults must actually corrupt the result
    assert changed > 40


def test_fault_is_upstream_of_the_answer():
    rng = random.Random(2)
    t = make_task(rng)
    assert t.fault_step < t.crash_step
    assert t.distance >= 1


def test_mock_respawn_recovers_retry_does_not():
    rng = random.Random(3)
    agent = MockAgent()
    crash_ok = respawn_ok = n = 0
    for _ in range(30):
        t = make_task(rng)
        plan = inject_fault(t, rng)
        wrong, _ = execute(t.table, plan)
        if verify(wrong, t):
            continue
        n += 1
        crash_ok += verify(agent.fix_answer(t, plan, wrong), t)
        fix = agent.fix_step(t, plan, t.fault_step)
        p2 = [dict(o) for o in plan]
        p2[t.fault_step] = fix
        ans2, _ = execute(t.table, p2)
        respawn_ok += verify(ans2, t)
    assert n > 20
    assert crash_ok == 0        # in-place re-answer (mock) never recovers
    assert respawn_ok == n      # redo-faulty-step (mock) always recovers