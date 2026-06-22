"""Does respawn decide WHERE to rewind — under genuine ambiguity?

A bug at step k propagates downstream, so steps k..4 all look anomalous. A real
(imperfect) heuristic attributor scores each step by how much *new* deviation it
introduced. We then compare three controllers, all driven by the real
respawn.recover(), under a budget too small to try every step:

  * retry-at-crash : only re-enter at the final step        -> can't fix upstream
  * blind          : recover() with a uniform attributor    -> re-enters at random
  * respawn        : recover() with the heuristic attributor -> re-enters at the guess

If respawn recovers more often than blind within the same budget, its
where-to-rewind decision is doing real work — the half of the claim the CSV demo
did not test. Fully offline.

    python examples/ambiguous_pipeline/run.py --trials 400
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "experiments"))

from _stats import assess  # noqa: E402
from stages import (  # noqa: E402
    CRASH_STEP,
    N_TRANSFORMS,
    Pipeline,
    make_trial,
    reference_means,
    validate,
)

from respawn import recover, uniform_attributor  # noqa: E402


def heuristic_posterior(pipe: Pipeline, ref_means, noise: float, rng) -> dict:
    """Real, imperfect attributor: score each step by the NEW deviation it adds
    over the previous step (the cause introduces the first jump; downstream steps
    mostly propagate it). Measurement noise makes it realistically fallible."""
    devs = []
    for i in range(N_TRANSFORMS):
        ref = ref_means[i]
        dev = abs(pipe.step_means[i] - ref) / (abs(ref) + 1.0)
        devs.append(dev)
    scores = []
    prev = 0.0
    for i in range(N_TRANSFORMS):
        incr = max(0.0, devs[i] - prev) + max(0.0, rng.gauss(0, noise))
        scores.append(incr)
        prev = devs[i]
    total = sum(scores) or 1.0
    return {i: scores[i] / total for i in range(N_TRANSFORMS)}


def make_reexecute(pipe: Pipeline, true_answer: float):
    def reexecute(from_step):
        if from_step >= N_TRANSFORMS:                 # crash step: nothing upstream fixed
            return validate(pipe, true_answer), []
        pipe.strategy[from_step] = 0                  # assume this step is the cause -> fix it
        pipe.run_from(from_step)                      # recompute forward
        return validate(pipe, true_answer), []
    return reexecute


def fresh(trial: Pipeline) -> Pipeline:
    p = Pipeline(trial.base_vals, buggy_step=trial.buggy_step).run()
    return p


def run_experiment(trials=400, budget=9, noise=1.2, explore=0.15, seed=0) -> dict:
    rng = random.Random(seed)
    retry_ok = blind_ok = respawn_ok = 0
    attr_top1 = 0
    b = c = 0                          # discordant pairs: respawn vs blind
    traj = list(range(CRASH_STEP + 1))  # length-6 placeholder records (steps 0..5)

    for _ in range(trials):
        trial = make_trial(rng)
        ref_means, true_ans = reference_means(trial.base_vals)

        post = heuristic_posterior(trial, ref_means, noise, rng)
        guess = max(post, key=post.get)
        attr_top1 += int(guess == trial.buggy_step)

        p = fresh(trial)
        r_retry = recover(traj, lambda t: {CRASH_STEP: 1.0},
                          make_reexecute(p, true_ans), budget=budget, explore=0.0)
        retry_ok += int(r_retry.recovered)

        p = fresh(trial)
        r_blind = recover(traj, uniform_attributor,
                          make_reexecute(p, true_ans), budget=budget,
                          explore=1.0, rng=random.Random(rng.random()))
        blind_hit = int(r_blind.recovered)
        blind_ok += blind_hit

        p = fresh(trial)
        post = heuristic_posterior(trial, ref_means, noise, rng)
        r_resp = recover(traj, lambda t, _p=post: _p,
                         make_reexecute(p, true_ans), budget=budget,
                         explore=explore, rng=random.Random(rng.random()))
        resp_hit = int(r_resp.recovered)
        respawn_ok += resp_hit

        if resp_hit and not blind_hit:
            b += 1
        elif blind_hit and not resp_hit:
            c += 1

    return {"trials": trials, "budget": budget, "noise": noise,
            "attr_acc": attr_top1 / trials, "retry": retry_ok / trials,
            "blind": blind_ok / trials, "respawn": respawn_ok / trials, "b": b, "c": c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--budget", type=int, default=9)
    ap.add_argument("--noise", type=float, default=1.2)
    ap.add_argument("--explore", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r = run_experiment(args.trials, args.budget, args.noise, args.explore, args.seed)
    print(f"ambiguous-attribution pipeline — {r['trials']} trials, "
          f"budget={r['budget']}, noise={r['noise']}\n")
    print(f"  heuristic attributor top-1 accuracy : {100*r['attr_acc']:5.1f}%  "
          f"(random = {100/N_TRANSFORMS:.0f}%)")
    print(f"  retry-at-crash recovery             : {100*r['retry']:5.1f}%")
    print(f"  blind re-entry recovery             : {100*r['blind']:5.1f}%")
    print(f"  respawn (where-guided) recovery      : {100*r['respawn']:5.1f}%")
    print("\n  respawn vs blind (does deciding WHERE help?):")
    print(assess(r["b"], r["c"], r["trials"], baseline="blind").line())


if __name__ == "__main__":
    main()