from recoverybench.controllers import (
    full_restart_controller,
    naive_truncate_controller,
    respawn_controller,
)
from recoverybench.model import RecoveryResult
from recoverybench.simulate import generate_scenario, run_benchmark, run_scenario


def test_generate_scenario_is_deterministic():
    s1 = generate_scenario(seed=42)
    s2 = generate_scenario(seed=42)
    assert s1.id == s2.id
    assert s1.archetype == s2.archetype
    assert s1.ground_truth_reentry == s2.ground_truth_reentry


def test_generate_scenario_different_seeds():
    s1 = generate_scenario(seed=1)
    s2 = generate_scenario(seed=2)
    assert s1.id != s2.id


def test_full_restart_controller_returns_result():
    scenario = generate_scenario(seed=0)
    result = run_scenario(scenario, full_restart_controller)
    assert isinstance(result, RecoveryResult)
    assert result.reentry_point == 0


def test_naive_truncate_reentry_at_fail_step():
    scenario = generate_scenario(seed=1)
    result = run_scenario(scenario, naive_truncate_controller)
    assert result.reentry_point == scenario.failure.step_index


def test_respawn_controller_returns_result():
    scenario = generate_scenario(seed=2)
    result = run_scenario(scenario, respawn_controller)
    assert isinstance(result, RecoveryResult)
    assert 0 <= result.reentry_point <= scenario.failure.step_index


def test_run_benchmark_covers_all_scenarios():
    scenarios = [generate_scenario(seed=i) for i in range(5)]
    controllers = {"respawn": respawn_controller}
    results = run_benchmark(scenarios, controllers)
    assert len(results["respawn"]) == 5


def test_wasted_call_ratio_range():
    scenario = generate_scenario(seed=10)
    result = run_scenario(scenario, respawn_controller)
    assert 0.0 <= result.wasted_call_ratio <= 1.0
