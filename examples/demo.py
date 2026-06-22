"""60-second demo. No API keys -- failures are simulated deterministically so
you can see respawn diagnose each one and retry *differently*.

    python -m demo      (from the repo root)  or   python demo.py
"""
from respawn import Escalate, FailureType, Policy, Respawn, WeakOutput


# --- Case 1: deterministic bad output. Replaying identically would loop forever.
# The "model" only emits the required field once the error is fed back to it.
def flaky_extractor(attempt):
    if "Fix exactly that" not in attempt.feedback:
        raise WeakOutput("missing required field 'invoice_total'")
    return {"invoice_total": 42.0, "model_used": attempt.model}


# --- Case 2: rate limit, then success once we've backed off.
_calls = {"n": 0}
def rate_limited_call(attempt):
    _calls["n"] += 1
    if _calls["n"] == 1:
        raise RuntimeError("429 Too Many Requests")
    return "ok"


# --- Case 3: hard problem -- needs to climb the model ladder to land.
def hard_reasoning(attempt):
    if attempt.model != "large":
        raise WeakOutput(f"answer from '{attempt.model}' failed the check")
    return f"correct answer (solved on {attempt.model})"


def show(title, step, policy=None, **kw):
    r = Respawn(budget=4, policy=policy, on_event=lambda m: None).run(step, **kw)
    print(f"\n=== {title} ===")
    print(r.explain())
    print("result:", r.result)


if __name__ == "__main__":
    show("bad output -> reframe with error fed back", flaky_extractor,
         prompt="extract invoice fields", model="small")
    show("rate limit -> back off, then succeed", rate_limited_call,
         prompt="call the API", model="small")

    # Custom policy: for weak answers, climb the model ladder immediately.
    escalate_first = Policy(ladders={FailureType.BAD_OUTPUT: [Escalate(), Escalate()]})
    show("weak answer -> escalate up the model ladder", hard_reasoning,
         policy=escalate_first, prompt="solve the hard task", model="small")