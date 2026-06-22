"""Live re-execution probe runner.

Measures, on real Who&When tasks, whether rewinding to the annotated decisive
step and re-running recovers the ground-truth answer more often than rewinding
only to the crash step -- and whether observed recovery tracks the analytic
fix_prob*(1-break)**d curve the rest of the repo assumes.

    # offline plumbing (no key) -- reproduces the analytic model BY CONSTRUCTION:
    python experiments/run_probe.py --provider mock --n 60 --trials 5

    # the real measurement (needs a key in your env):
    python experiments/run_probe.py --provider anthropic --model claude-haiku-4-5 \
        --n 40 --trials 3 --data "Agents_Failure_Attribution/Who&When"

Honest scope: continuation is single-model (an approximation of the original
multi-agent run). The ABSOLUTE recovery numbers depend on that procedure; the
COMPARISON (cause vs crash, identical procedure) is what the probe validates.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe import (  # noqa: E402
    ProbeTask,
    attempt_recover,
    make_mock_continuation,
    make_real_continuation,
)

FIX_PROB = 0.5
RELIABILITY = 0.85


def load_probe_tasks(data_dir, limit, seed=0):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"No Who&When json under {data_dir!r}. Clone "
                         "github.com/ag2ai/Agents_Failure_Attribution first.")
    tasks = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        hist = d.get("history", [])
        gt = str(d.get("ground_truth", "")).strip()
        try:
            step = int(str(d.get("mistake_step", "")).strip())
        except ValueError:
            continue
        if len(hist) < 2 or not gt or step < 0 or step >= len(hist):
            continue
        tasks.append(ProbeTask(
            id=str(d.get("question_ID", os.path.basename(f))),
            question=str(d.get("question", "")),
            ground_truth=gt, history=hist, root=step,
            system_prompt=str(d.get("system_prompt", "")),
        ))
    random.Random(seed).shuffle(tasks)
    return tasks[:limit]


def build_continuation(args):
    if args.provider == "mock":
        return make_mock_continuation(FIX_PROB, RELIABILITY, seed=1), "MOCK (plumbing only)"
    if args.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
    elif args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
    else:
        raise SystemExit(f"unknown provider {args.provider!r}")
    return (make_real_continuation(client, args.model, max_chars=args.max_chars),
            f"REAL (provider={args.provider}, model={args.model})")


def _rate(pair):
    h, n = pair
    return (h / n) if n else float("nan")


def measure(tasks, cont, trials, rng):
    """Run `trials` re-entries at the crash step and at the cause, per task.

    Split by distance bucket, because the comparison can only DIFFER on d>=1:
      * d == 0  -> cause IS the crash step; the two strategies are identical.
      * d >= 1  -> the real test (retry-at-crash re-runs the wrong-context tail;
                   rewind-to-cause gets a cleaner slate at the actual cause).
    """
    # bucket -> {"crash": [hits, n], "cause": [hits, n], "tasks": set, "b":, "c":}
    buckets = {b: {"crash": [0, 0], "cause": [0, 0], "tasks": set(), "b": 0, "c": 0}
               for b in ("d=0", "d>=1", "all")}
    by_distance = {}
    for t in tasks:
        b = "d=0" if t.distance == 0 else "d>=1"
        for _ in range(trials):
            c_ok = int(attempt_recover(t, t.crash, cont))
            k_ok = int(attempt_recover(t, t.root, cont))
            for key in (b, "all"):
                buckets[key]["crash"][0] += c_ok
                buckets[key]["crash"][1] += 1
                buckets[key]["cause"][0] += k_ok
                buckets[key]["cause"][1] += 1
                buckets[key]["tasks"].add(t.id)
                if k_ok and not c_ok:
                    buckets[key]["b"] += 1
                elif c_ok and not k_ok:
                    buckets[key]["c"] += 1
            slot = by_distance.setdefault(t.distance, [0, 0])
            slot[0] += k_ok
            slot[1] += 1
    out = {}
    for b, d in buckets.items():
        out[b] = {"retry_at_crash": _rate(d["crash"]),
                  "rewind_to_cause": _rate(d["cause"]),
                  "n_tasks": len(d["tasks"]),
                  "n_pairs": d["crash"][1], "b": d["b"], "c": d["c"]}
    out["by_distance"] = {d: h / n for d, (h, n) in sorted(by_distance.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--n", type=int, default=60, help="number of tasks")
    ap.add_argument("--trials", type=int, default=5, help="re-runs per re-entry point")
    ap.add_argument("--max-chars", type=int, default=400,
                    help="cap chars per transcript message (lower = cheaper)")
    ap.add_argument("--data", default=os.environ.get(
        "WHOWHEN_DIR", "/home/claude/Agents_Failure_Attribution/Who&When"))
    args = ap.parse_args()

    tasks = load_probe_tasks(args.data, args.n)
    cont, label = build_continuation(args)
    rng = random.Random(7)
    print(f"Probe: {label}")
    print(f"{len(tasks)} tasks x {args.trials} trials per re-entry point\n")

    res = measure(tasks, cont, args.trials, rng)

    def fmt(x):
        return "  n/a" if x != x else f"{100*x:4.1f}%"   # x!=x catches NaN

    print("Observed recovery, split by how far upstream the cause is:")
    print(f"{'bucket':8s} {'tasks':>6s} {'retry-at-crash':>16s} {'rewind-to-cause':>17s}")
    for b in ("d=0", "d>=1", "all"):
        r = res[b]
        print(f"{b:8s} {r['n_tasks']:6d} {fmt(r['retry_at_crash']):>16s} "
              f"{fmt(r['rewind_to_cause']):>17s}")
    test = res["d>=1"]
    print(f"\nTHE TEST is the d>=1 row ({test['n_tasks']} tasks): at d>=1 the cause is "
          "upstream of the crash step,")
    print("so this is the only regime where rewind-to-cause can beat retry-at-crash.")
    from _stats import assess  # experiments/ is on sys.path
    v = assess(test["b"], test["c"], test["n_pairs"])
    print(v.line())
    print()

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    import numpy as np
    groups = ["d=0\n(strategies identical)", "d>=1\n(the real test)", "all"]
    crash_vals = [res[b]["retry_at_crash"] for b in ("d=0", "d>=1", "all")]
    cause_vals = [res[b]["rewind_to_cause"] for b in ("d=0", "d>=1", "all")]
    crash_vals = [0 if v != v else v for v in crash_vals]
    cause_vals = [0 if v != v else v for v in cause_vals]
    x = np.arange(len(groups)); w = 0.38
    ax[0].bar(x - w/2, crash_vals, w, color="#c0392b", label="retry-at-crash")
    ax[0].bar(x + w/2, cause_vals, w, color="#16a085", label="rewind-to-cause")
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, fontsize=8)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("observed recovery rate")
    ax[0].set_title("Observed recovery by distance bucket")
    ax[0].legend(fontsize=8, frameon=False)

    ds = sorted(res["by_distance"])
    obs = [res["by_distance"][d] for d in ds]
    curve = [FIX_PROB * (RELIABILITY) ** d for d in ds]
    ax[1].plot(ds, obs, "o", ms=7, color="#16a085", label="observed (rewind-to-cause)")
    ax[1].plot(ds, curve, "--", color="#8e44ad",
               label=f"analytic {FIX_PROB}*{RELIABILITY}^d")
    ax[1].set_xlabel("distance d = crash - cause")
    ax[1].set_ylabel("recovery rate"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Does observed recovery track the analytic curve?")
    ax[1].legend(fontsize=8, frameon=False)

    fig.suptitle(f"Live re-execution probe  [{label}]", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "images", "probe_results")
    fig.savefig(out + ".png", bbox_inches="tight")
    print(f"saved figure -> {out}.png")


if __name__ == "__main__":
    main()