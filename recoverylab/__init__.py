"""recoverylab -- controlled pipeline tasks where rollback is necessary.

The regime respawn's thesis actually depends on: a silent upstream fault that
re-answering in place cannot fix, only redoing the faulty step + recomputing.
"""
from .agent import MockAgent, RealAgent
from .engine import Task, execute, inject_fault, make_task, verify

__all__ = [
    "Task", "make_task", "execute", "inject_fault", "verify",
    "MockAgent", "RealAgent",
]