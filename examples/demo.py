"""Demo: Respawn recovery without an API key.

Shows the core loop: simulate a failure, classify it, get a recovery plan.

    python examples/demo.py
"""
from respawn import Respawn
from respawn.failures import FailureEvent, FailureKind

# A synthetic 10-step agent trace
trace = [
    {"index": i, "role": "agent" if i % 2 == 0 else "tool", "content": f"step {i}"}
    for i in range(10)
]

rs = Respawn()

print("=== Transient failure (timeout at step 9) ===")
event = FailureEvent(kind=FailureKind.TRANSIENT, step_index=9, message="Connection timeout")
plan = rs.recover(failure=event, trace=trace)
print(f"  strategy      : {plan.strategy}")
print(f"  reentry_point : {plan.reentry_point}")
print(f"  patch         : {plan.patch}")
print(f"  rationale     : {plan.rationale}")
print()

print("=== Semantic failure (drift detected at step 7, cause at step 4) ===")
event = FailureEvent(kind=FailureKind.SEMANTIC, step_index=7, message="Off-topic response detected")
plan = rs.recover(failure=event, trace=trace)
print(f"  strategy      : {plan.strategy}")
print(f"  reentry_point : {plan.reentry_point}")
print(f"  patch         : {plan.patch}")
print(f"  rationale     : {plan.rationale}")
print()

print("=== Policy failure ===")
event = FailureEvent(kind=FailureKind.POLICY, step_index=6, message="Content policy violation")
plan = rs.recover(failure=event, trace=trace)
print(f"  strategy      : {plan.strategy}")
print(f"  reentry_point : {plan.reentry_point}")
print()

print("=== Exception-based recovery (auto-classify) ===")
try:
    raise TimeoutError("Connection timed out after 30s")
except Exception as exc:
    plan = rs.recover(failure=exc, trace=trace)
    print(f"  classified as : {plan.strategy}")
    print(f"  rationale     : {plan.rationale}")
