"""Run RecoveryBench and write results + figures.

Usage:
    python experiments/run_recoverybench.py --scenarios data/scenarios.jsonl \
        --output results/ --figures docs/images/

    # Generate scenarios only (no API key needed):
    python experiments/run_recoverybench.py --generate-only --output data/scenarios.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def generate_scenarios(n: int = 500) -> list:
    from recoverybench.simulate import generate_scenario

    return [generate_scenario(seed=i) for i in range(n)]


def save_scenarios(scenarios: list, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in scenarios:
            record = {
                "id": s.id,
                "archetype": s.archetype,
                "failure_kind": s.failure.kind,
                "fail_step": s.failure.step_index,
                "ground_truth_reentry": s.ground_truth_reentry,
                "ground_truth_strategy": s.ground_truth_strategy,
                "n_steps": len(s.trace),
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(scenarios)} scenarios to {path}")


def run_benchmark(scenarios: list, results_dir: pathlib.Path) -> dict:
    from recoverybench.controllers import (
        full_restart_controller,
        naive_truncate_controller,
        respawn_controller,
    )
    from recoverybench.simulate import run_benchmark as _run

    controllers = {
        "full_restart": full_restart_controller,
        "naive_truncate": naive_truncate_controller,
        "respawn": respawn_controller,
    }
    results = _run(scenarios, controllers)

    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, res_list in results.items():
        tcr = sum(r.task_completed for r in res_list) / len(res_list)
        wcr = sum(r.wasted_call_ratio for r in res_list) / len(res_list)
        summary[name] = {"tcr": round(tcr, 4), "wcr": round(wcr, 4), "n": len(res_list)}

    out = results_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def plot_figures(summary: dict, figures_dir: pathlib.Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures", file=sys.stderr)
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    names = list(summary.keys())
    tcrs = [summary[n]["tcr"] * 100 for n in names]
    wcrs = [summary[n]["wcr"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, tcrs)
    ax1.set_ylabel("Task Completion Rate (%)")
    ax1.set_title("TCR by strategy")

    ax2.bar(names, wcrs)
    ax2.set_ylabel("Wasted-Call Ratio")
    ax2.set_title("WCR by strategy (lower is better)")

    plt.tight_layout()
    out = figures_dir / "recoverybench_results.png"
    plt.savefig(out, dpi=150)
    print(f"Saved figure to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=pathlib.Path, default=pathlib.Path("data/scenarios.jsonl"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/"))
    parser.add_argument("--figures", type=pathlib.Path, default=pathlib.Path("docs/images/"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    scenarios = generate_scenarios(n=args.n)

    if args.generate_only:
        save_scenarios(scenarios, args.scenarios)
        return

    summary = run_benchmark(scenarios, args.output)
    plot_figures(summary, args.figures)


if __name__ == "__main__":
    main()
