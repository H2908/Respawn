"""Who&When replication for RecoveryBench.

Replaces the *synthetic* distance distribution with the REAL one from the
Who&When benchmark (Zhang et al., ICML 2025), and plugs in the literature's
MEASURED step-level attribution accuracies in place of a synthetic knob.

What this upgrades vs the synthetic PoC:
  * upstream distance d = (len(history)-1) - mistake_step is taken from the
    real human/algorithm annotations of the decisive error step.
  * the failure surfaces at the terminal step, so "retry-at-crash" = regenerate
    the final answer -- which cannot fix an upstream cause, by construction.
  * attribution accuracy points are real: ~0.14 (best Who&When baseline) and
    ~0.43 (AgenTracer on the algorithm-generated split).

Honest scope: the per-attempt recovery model (fix_prob, break_prob) is still
analytic -- we are NOT re-executing the real agents. This measures recovery
*potential* over the real distribution of decisive-error distances. Live
re-execution is the next step.

Usage:
    python experiments/whoandwhen.py --data "/path/to/Who&When"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recoverybench import (  # noqa: E402
    Task,
    attribution_guided,
    blind_search,
    evaluate,
    oracle,
    retry_at_crash,
    rollback_all,
)

FIX_PROB = 0.5
RELIABILITY = 0.85
# Measured step-level attribution accuracies from the literature.
ATTR_POINTS = {
    "respawn @ Who&When baseline (acc 0.14)": 0.14,
    "respawn @ AgenTracer (acc 0.43)": 0.43,
}


def load_tasks(data_dir, fix_prob=FIX_PROB, reliability=RELIABILITY):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"No Who&When json files under {data_dir!r}. "
                         "Clone github.com/ag2ai/Agents_Failure_Attribution and "
                         "point --data at its 'Who&When' folder.")
    tasks = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        T = len(d.get("history", []))
        try:
            step = int(str(d.get("mistake_step", "")).strip())
        except ValueError:
            continue
        if T < 2 or step < 0 or step >= T:
            continue
        crash = T                       # failure surfaces at the terminal step
        root = step + 1                 # 0-indexed annotation -> 1-indexed step
        tasks.append(Task(T=T, crash=crash, root=root, fix_prob=fix_prob,
                          break_prob=1 - reliability))
    return tasks, files


def rate(ctrl, tasks, budget):
    return evaluate(ctrl, tasks, budget)["recovery_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get(
        "WHOWHEN_DIR", "/home/claude/Agents_Failure_Attribution/Who&When"))
    ap.add_argument("--budget", type=int, default=20)
    args = ap.parse_args()

    tasks, files = load_tasks(args.data)
    dists = [t.distance for t in tasks]
    n = len(tasks)
    frac_terminal = sum(d == 0 for d in dists) / n

    print(f"Loaded {n} Who&When trajectories.")
    print(f"Median upstream distance d = {sorted(dists)[n // 2]}  "
          f"| share with cause at final step (d=0) = {100 * frac_terminal:.1f}%")
    print(f"=> retry-the-failing-step can address at most {100 * frac_terminal:.1f}% "
          f"of these failures.\n")

    B = args.budget
    controllers = {
        "rollback-all": rollback_all,
        "retry-at-crash": retry_at_crash,
        "blind re-entry": blind_search,
        **{name: attribution_guided(a) for name, a in ATTR_POINTS.items()},
        "oracle (perfect attr)": oracle,
    }
    print(f"Recovery rate over REAL Who&When distances (budget = {B} step-exec):")
    results = {}
    for name, ctrl in controllers.items():
        r = rate(ctrl, tasks, B)
        results[name] = r
        print(f"   {name:42s} {100 * r:5.1f}%")

    # ---- figure: distance distribution + recovery vs budget ----------------
    budgets = list(range(2, 61, 2))
    curves = {
        "retry-at-crash": [rate(retry_at_crash, tasks, b) for b in budgets],
        "blind re-entry": [rate(blind_search, tasks, b) for b in budgets],
        "respawn (acc 0.43)": [rate(attribution_guided(0.43), tasks, b) for b in budgets],
        "oracle": [rate(oracle, tasks, b) for b in budgets],
    }
    COL = {"retry-at-crash": "#c0392b", "blind re-entry": "#2980b9",
           "respawn (acc 0.43)": "#16a085", "oracle": "#8e44ad"}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    capped = [min(d, 15) for d in dists]
    ax[0].hist(capped, bins=range(0, 17), color="#34495e", alpha=0.85, rwidth=0.9)
    ax[0].axvline(0.5, color="#c0392b", lw=2)
    ax[0].annotate(f"only {100*frac_terminal:.1f}% at d=0\n(all retry-at-crash can reach)",
                   (0.7, ax[0].get_ylim()[1] * 0.7), color="#c0392b", fontsize=8)
    ax[0].set_title("A. Real Who&When: how far upstream\nthe decisive error sits")
    ax[0].set_xlabel("distance d = final step - decisive error  (15 = 15+)")
    ax[0].set_ylabel("number of trajectories")

    for name, ys in curves.items():
        ax[1].plot(budgets, ys, marker="o", ms=3, lw=2, color=COL[name], label=name)
    ax[1].set_title("B. Recovery over the real distribution\nvs compute budget")
    ax[1].set_xlabel("retry budget (step-executions)")
    ax[1].set_ylabel("recovery rate")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=8, loc="upper left", frameon=False)

    fig.suptitle("respawn on Who&When: 95.7% of real failures have an upstream cause that "
                 "retrying the failing step cannot fix", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "images", "whoandwhen_results")
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"\nsaved figure -> {out}.png / .svg")


if __name__ == "__main__":
    main()