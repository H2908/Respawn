"""Demo: Respawn with a live LLM agent (requires ANTHROPIC_API_KEY).

Runs a minimal 3-step agent, injects a synthetic failure at step 2,
and shows Respawn recovering it.

    export ANTHROPIC_API_KEY=sk-...
    python examples/demo_llm.py
"""
from __future__ import annotations

import os

from respawn import Respawn
from respawn.failures import FailureEvent, FailureKind
from respawn.llm import complete

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")

SYSTEM = "You are a helpful assistant. Answer in one sentence."

rs = Respawn()
trace: list[dict] = []


def agent_step(messages: list[dict], index: int) -> str:
    reply = complete(messages)
    trace.append({"index": index, "role": "agent", "content": reply})
    return reply


print("--- Running agent steps ---")
messages = [{"role": "user", "content": "What is the capital of France?"}]
reply = agent_step(messages, index=0)
print(f"Step 0: {reply}")

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "And what is its population?"})
reply = agent_step(messages, index=1)
print(f"Step 1: {reply}")

print("\n--- Injecting transient failure at step 2 ---")
failure = FailureEvent(kind=FailureKind.TRANSIENT, step_index=2, message="Simulated timeout")
plan = rs.recover(failure=failure, trace=trace)
print(f"Recovery plan: {plan}")

print(f"\n--- Re-entering at step {plan.reentry_point} with patch {plan.patch} ---")
messages.append({"role": "user", "content": "Describe Paris in one sentence."})
reply = agent_step(messages, index=2)
print(f"Step 2 (recovered): {reply}")
