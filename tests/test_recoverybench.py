"""Honesty guardrails for RecoveryBench, enforced in CI.

If any of these break, the benchmark is rigged. A PR that makes respawn look
better by breaking one of these should be closed.
"""
from recoverybench import (
    attribution_guided,
    blind_search,
    evaluate,
    make_pool,
    oracle,
    retry_at_crash,
)
from recoverybench.model import Task

POOL = make_pool(4000, seed=0, T=12, reliability=0.85, fix_prob=0.5,
                 transient_fraction=0.3, max_distance=6)
B = 20


def _rate(ctrl):
    return evaluate(ctrl, POOL, B)["recovery_rate"]


def test_guided_perfect_attribution_equals_oracle():
    assert abs(_rate(attribution_guided(1.0)) - _rate(oracle)) < 0.03


def test_guided_uninformed_equals_blind_search():
    # respawn can NEVER beat blind search when attribution carries no information
    assert abs(_rate(attribution_guided(0.0)) - _rate(blind_search)) < 0.03


def test_retry_at_crash_cannot_recover_upstream_causes():
    upstream = [Task(T=12, crash=10, root=7, fix_prob=0.5, break_prob=0.15)
                for _ in range(2000)]
    assert evaluate(retry_at_crash, upstream, B)["recovery_rate"] == 0.0


def test_retry_at_crash_recovers_transient_failures():
    transient = [Task(T=12, crash=10, root=10, fix_prob=0.5, break_prob=0.15)
                 for _ in range(2000)]
    assert evaluate(retry_at_crash, transient, B)["recovery_rate"] > 0.5


def test_respawn_beats_retry_at_crash_at_realistic_attribution():
    # at ~0.30 step accuracy (within what real attributors achieve)
    assert _rate(attribution_guided(0.30)) > _rate(retry_at_crash)


def test_monotonic_in_attribution_accuracy():
    rates = [_rate(attribution_guided(a)) for a in (0.0, 0.3, 0.6, 1.0)]
    assert rates == sorted(rates)