"""RecoveryBench -- measuring recovery lift, not attribution accuracy."""
from .controllers import (
    attribution_guided,
    blind_search,
    oracle,
    retry_at_crash,
    rollback_all,
)
from .model import Task, sample_task
from .simulate import evaluate, make_pool

__all__ = [
    "Task", "sample_task", "make_pool", "evaluate",
    "retry_at_crash", "rollback_all", "blind_search", "attribution_guided", "oracle",
]