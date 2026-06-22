"""Run RecoveryBench: sanity checks first, then the three sweeps + figure.

    python experiments/run_recoverybench.py
"""
from __future__ import annotations

import os
import random
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
    make_pool,
    oracle,
    retry_at_crash,
    rollback_all,
)

# ---- fixed experimental settings -------------------------------------------
T = 12
RELIABILITY = 0.85          # ~0.85 per-step reliability, from the problem statement
FIX_PROB = 0.5              # a different retry fixes a genuinely-fixable step ~50%
TRANSIENT = 0.30            # 30% of failures are plain transients (d = 0)
MAXD = 6
N = 6000                    # tasks per pool (low Monte Carlo noise)

POOL_KW = dict(T=T, reliability=RELIABILITY, fix_prob=FIX_PROB,
               transient_fraction=TRANSIENT, max_distance=MAXD)


def fixed_distance_pool(n, d, seed=0):
    """Pool where every task has a fixed upstream distance d (d=0 => transient)."""
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        crash = rng.randint(max(3, T - 2), T)
        root = crash - min(d, crash - 1)
        tasks.append(Task(T=T, crash=crash, root=root, fix_prob=FIX_PROB,
                          break_prob=1 - RELIABILITY))
    return tasks


def pct(x):
    return f"{100 * x:5.1f}%"


def sanity_checks(pool):
    print("=" * 64)
    print("HONESTY SANITY CHECKS (must all hold or the benchmark is rigged)")
    print("=" * 64)
    B = 20
    guided1 = evaluate(attribution_guided(1.0), pool, B)["recovery_rate"]
    orac = evaluate(oracle, pool, B)["recovery_rate"]
    guided0 = evaluate(attribution_guided(0.0), pool, B)["recovery_rate"]
    blind = evaluate(blind_search, pool, B)["recovery_rate"]

    print(f"1. guided(acc=1.0) == oracle           : "
          f"{pct(guided1)} vs {pct(orac)}   "
          f"{'OK' if abs(guided1 - orac) < 0.02 else 'FAIL'}")
    print(f"2. guided(acc=0, no info) == blind     : "
          f"{pct(guided0)} vs {pct(blind)}   "
          f"{'OK' if abs(guided0 - blind) < 0.02 else 'FAIL'}")

    # 3. retry-at-crash recovers transients, ~0 on upstream causes.
    up = fixed_distance_pool(N, d=3, seed=7)
    tr = fixed_distance_pool(N, d=0, seed=7)
    r_up = evaluate(retry_at_crash, up, B)["recovery_rate"]
    r_tr = evaluate(retry_at_crash, tr, B)["recovery_rate"]
    print(f"3. retry-at-crash: upstream ~0, transient > 0 : "
          f"up={pct(r_up)}  transient={pct(r_tr)}   "
          f"{'OK' if r_up < 0.01 and r_tr > 0.5 else 'FAIL'}")
    print()


def sweep_budget(pool, budgets):
    rows = {}
    controllers = {
        "retry-at-crash": retry_at_crash,
        "rollback-all": rollback_all,
        "blind re-entry": blind_search,
        "respawn (attr acc=0.5)": attribution_guided(0.5),
        "oracle (perfect attr)": oracle,
    }
    for name, ctrl in controllers.items():
        rows[name] = [evaluate(ctrl, pool, b)["recovery_rate"] for b in budgets]
    return rows


def sweep_accuracy(pool, accs, budget):
    guided = [evaluate(attribution_guided(a), pool, budget)["recovery_rate"] for a in accs]
    blind = evaluate(blind_search, pool, budget)["recovery_rate"]
    crash = evaluate(retry_at_crash, pool, budget)["recovery_rate"]
    orac = evaluate(oracle, pool, budget)["recovery_rate"]
    return guided, blind, crash, orac


def sweep_distance(distances, budget):
    out = {"retry-at-crash": [], "blind re-entry": [],
           "respawn (acc=0.5)": [], "oracle": []}
    for d in distances:
        pool = fixed_distance_pool(N, d=d, seed=11)
        out["retry-at-crash"].append(evaluate(retry_at_crash, pool, budget)["recovery_rate"])
        out["blind re-entry"].append(evaluate(blind_search, pool, budget)["recovery_rate"])
        out["respawn (acc=0.5)"].append(evaluate(attribution_guided(0.5), pool, budget)["recovery_rate"])
        out["oracle"].append(evaluate(oracle, pool, budget)["recovery_rate"])
    return out


def main():
    pool = make_pool(N, seed=0, **POOL_KW)
    sanity_checks(pool)

    budgets = list(range(2, 41, 2))
    budget_rows = sweep_budget(pool, budgets)

    accs = [i / 20 for i in range(0, 21)]
    BUD = 20
    guided_acc, blind_h, crash_h, orac_h = sweep_accuracy(pool, accs, BUD)

    dists = list(range(0, 7))
    dist_rows = sweep_distance(dists, BUD)

    # ---- print headline table ----------------------------------------------
    print("Recovery rate at budget = 20 step-executions (mixed pool):")
    for name, ys in budget_rows.items():
        b20 = ys[budgets.index(20)]
        print(f"   {name:26s} {pct(b20)}")
    print()

    # ---- figure -------------------------------------------------------------
    COL = {
        "retry-at-crash": "#c0392b", "rollback-all": "#7f8c8d",
        "blind re-entry": "#2980b9", "respawn (attr acc=0.5)": "#16a085",
        "oracle (perfect attr)": "#8e44ad",
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: recovery vs budget
    for name, ys in budget_rows.items():
        ax[0].plot(budgets, ys, marker="o", ms=3, lw=2,
                   color=COL[name], label=name)
    ax[0].set_title("A. Recovery vs compute budget\n(mixed failures)")
    ax[0].set_xlabel("retry budget (step-executions)")
    ax[0].set_ylabel("recovery rate")
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8, loc="upper left", frameon=False)

    # Panel B: recovery vs attribution accuracy
    ax[1].plot(accs, guided_acc, marker="o", ms=3, lw=2.4, color="#16a085",
               label="respawn (attr-guided)")
    ax[1].axhline(blind_h, ls="--", lw=1.6, color="#2980b9", label="blind re-entry")
    ax[1].axhline(crash_h, ls="--", lw=1.6, color="#c0392b", label="retry-at-crash")
    ax[1].axhline(orac_h, ls=":", lw=1.6, color="#8e44ad", label="oracle")
    # where respawn first clears blind search
    cross = next((accs[i] for i in range(len(accs)) if guided_acc[i] > blind_h + 0.02), None)
    if cross is not None:
        ax[1].axvline(cross, color="#16a085", lw=1, alpha=0.4)
        ax[1].annotate(f"clears blind\nby attr acc {cross:.2f}", (cross, blind_h),
                       textcoords="offset points", xytext=(10, -30), fontsize=8,
                       color="#16a085")
    ax[1].set_title("B. Recovery vs attribution accuracy\n(budget = 20)")
    ax[1].set_xlabel("attribution accuracy (P pointing at true cause)")
    ax[1].set_ylabel("recovery rate")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=8, loc="upper left", frameon=False)

    # Panel C: recovery vs upstream distance
    dc = {"retry-at-crash": "#c0392b", "blind re-entry": "#2980b9",
          "respawn (acc=0.5)": "#16a085", "oracle": "#8e44ad"}
    for name, ys in dist_rows.items():
        ax[2].plot(dists, ys, marker="o", ms=4, lw=2, color=dc[name], label=name)
    ax[2].set_title("C. Recovery vs how far upstream\nthe cause is (budget = 20)")
    ax[2].set_xlabel("distance d = crash - root cause")
    ax[2].set_ylabel("recovery rate")
    ax[2].set_ylim(0, 1)
    ax[2].legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle("RecoveryBench: attribution-guided re-entry recovers runs that "
                 "retrying the failing step cannot \u2014 well below perfect attribution",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "images", "recoverybench_results")
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".svg", bbox_inches="tight")
    print(f"saved figure -> {out}.png / .svg")


if __name__ == "__main__":
    main()