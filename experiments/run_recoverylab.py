"""Controlled recovery experiment: does redoing the faulty step beat re-answering?

Each task: build a pipeline with a known answer, inject a silent upstream fault,
confirm the run now fails, then try to recover two ways with the SAME info:
  * retry-at-crash : ask the model for the corrected final answer (one shot).
  * respawn        : ask the model to fix the faulty step; recompute forward.

    # offline plumbing (no key):
    python experiments/run_recoverylab.py --provider mock --n 30

    # the real measurement (needs a key):
    python experiments/run_recoverylab.py --provider anthropic --model claude-haiku-4-5 --n 30
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recoverylab import (  # noqa: E402
    MockAgent, RealAgent, execute, inject_fault, make_task, verify,
)


def build_agent(args):
    if args.provider == "mock":
        return MockAgent()
    if args.provider == "anthropic":
        from anthropic import Anthropic
        return RealAgent(Anthropic(), args.model)
    if args.provider == "openai":
        from openai import OpenAI
        return RealAgent(OpenAI(), args.model)
    raise SystemExit(f"unknown provider {args.provider!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    agent = build_agent(args)
    rng = random.Random(args.seed)
    print(f"recoverylab: {agent.label}")

    crash_ok = respawn_ok = usable = 0
    disc_b = disc_c = 0          # discordant pairs for McNemar
    for _ in range(args.n):
        task = make_task(rng, n_rows=args.rows)
        plan = inject_fault(task, rng)
        wrong, _ = execute(task.table, plan)
        if verify(wrong, task):           # fault didn't actually change the answer
            continue                       # skip non-discriminating task
        usable += 1

        # retry-at-crash: re-answer in place
        ans = agent.fix_answer(task, plan, wrong)
        c_ok = int(verify(ans, task))
        crash_ok += c_ok

        # respawn: redo the faulty step, recompute forward (deterministic)
        fix = agent.fix_step(task, plan, task.fault_step)
        r_ok = 0
        if fix is not None:
            plan2 = [dict(op) for op in plan]
            plan2[task.fault_step] = fix
            ans2, _ = execute(task.table, plan2)
            r_ok = int(verify(ans2, task))
        respawn_ok += r_ok

        if r_ok and not c_ok:
            disc_b += 1
        elif c_ok and not r_ok:
            disc_c += 1

    crash_rate = crash_ok / usable if usable else 0.0
    respawn_rate = respawn_ok / usable if usable else 0.0
    print(f"\n{usable} usable tasks (silent upstream fault that changed the answer)\n")
    print(f"   retry-at-crash (re-answer in place) : {100*crash_rate:5.1f}%")
    print(f"   respawn (redo faulty step + recompute): {100*respawn_rate:5.1f}%")
    from _stats import assess  # experiments/ is on sys.path
    print(assess(disc_b, disc_c, usable).line())

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(["retry-at-crash\n(re-answer)", "respawn\n(redo faulty step)"],
           [crash_rate, respawn_rate], color=["#c0392b", "#16a085"])
    for i, v in enumerate([crash_rate, respawn_rate]):
        ax.text(i, v + 0.02, f"{100*v:.0f}%", ha="center", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("recovery rate")
    ax.set_title(f"Controlled pipeline w/ silent upstream fault\n[{agent.label}], "
                 f"{usable} tasks")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "images", "recoverylab_results")
    fig.savefig(out + ".png", bbox_inches="tight")
    print(f"\nsaved figure -> {out}.png")


if __name__ == "__main__":
    main()