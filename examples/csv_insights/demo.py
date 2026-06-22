"""CSV insights pipeline — a real, offline showcase of respawn.recover().

A messy sales CSV is run through load → parse_amount → filter → aggregate →
summarize. The default parse strategy silently can't read amounts like
"$1,200.00" and zeroes them, so the reported total is badly wrong WITHOUT
crashing. We then compare two recoveries:

  * retry-at-crash : re-run the final summary step  -> still wrong (cause is upstream)
  * respawn        : recover() re-enters at the cause, switches parse strategy,
                     and recomputes forward -> correct

Run:  python examples/csv_insights/demo.py     (no API key, fully deterministic)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from respawn import point_attributor, recover  # noqa: E402

from pipeline import build_pipeline, validate  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sales.csv")
TRUE_TOTAL = 16116.88        # correct complete-revenue (robust parse), for display


def show_summary(label, state):
    print(f"  {label}: total = ${state.get('total', 0):,.2f}   "
          f"top = {state.get('top_category')}   "
          f"quality = {'OK' if validate(state) else 'LOW (untrustworthy)'}")


def localize(trajectory):
    """Heuristic attributor: the earliest step that pushed data quality below the
    gate is the likely cause. Returns its step index."""
    for entry in trajectory:
        if entry["good_fraction"] < 0.9:
            return entry["step"]
    return trajectory[-1]["step"]


def main():
    print("=" * 70)
    print("CSV insights pipeline — recovering a silent upstream parse fault")
    print("=" * 70)

    # 1) Run the pipeline with default strategies.
    pipe = build_pipeline(CSV)
    traj = pipe.run()
    print("\nPipeline trace (good_fraction = parsed amounts that are non-zero):")
    for e in traj:
        print(f"  step {e['step']} {e['name']:<16} [{e['strategy']:<12}] "
              f"good_fraction={e['good_fraction']}")
    print()
    show_summary("initial run ", pipe.final)
    print(f"  (the true total is ${TRUE_TOTAL:,.2f} — the run is off by "
          f"${TRUE_TOTAL - pipe.final['total']:,.2f})\n")

    if validate(pipe.final):
        print("Validation passed — nothing to recover.")
        return

    # 2) Baseline: retry-at-crash = re-run ONLY the final step. Cause is upstream,
    #    so this changes nothing.
    pipe_b = build_pipeline(CSV)
    pipe_b.run()
    pipe_b.run_from(len(pipe_b.steps) - 1)     # re-run just `summarize`
    print("retry-at-crash (re-run the summary step):")
    show_summary("after retry ", pipe_b.final)
    print("  -> still wrong: re-running the summary can't un-corrupt upstream data.\n")

    # 3) respawn.recover(): attribute the cause, re-enter there, retry differently.
    pipe_r = build_pipeline(CSV)
    traj = pipe_r.run()
    suspect = localize(traj)
    print(f"respawn: attributor points at step {suspect} "
          f"('{pipe_r.steps[suspect].name}') as the cause.")

    def reexecute(from_step):
        step = pipe_r.steps[from_step]
        if step.idx + 1 >= len(step.strategies):
            return False, traj                 # no other strategy to try
        step.idx += 1                          # retry that step DIFFERENTLY
        new_traj = pipe_r.run_from(from_step)  # recompute forward
        return validate(pipe_r.final), new_traj

    res = recover(traj, point_attributor(suspect, 0.9), reexecute, budget=10, explore=0.0)
    print(res.explain())
    show_summary("after respawn", pipe_r.final)
    ok = abs(pipe_r.final["total"] - TRUE_TOTAL) < 0.01
    print(f"\n  -> recovered the correct total: {'YES' if ok else 'NO'} "
          f"(${pipe_r.final['total']:,.2f})")


if __name__ == "__main__":
    main()