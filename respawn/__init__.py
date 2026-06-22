"""respawn -- adaptive recovery for AI agents.

Durable execution resumes the same failing step. respawn retries it differently.
"""
from .core import Attempt, Receipt, Respawn, Trace
from .failures import (
    FailureInfo,
    FailureType,
    ToolError,
    WeakOutput,
    default_classifier,
)
from .llm import ChatResult, respawn_chat
from .policy import Policy, default_ladders
from .recover import (
    RecoveryResult,
    point_attributor,
    recover,
    uniform_attributor,
)
from .strategies import Backoff, Decompose, Escalate, Reframe, SwapTool

__version__ = "0.1.0"
__all__ = [
    "Respawn", "Attempt", "Receipt", "Trace",
    "FailureType", "FailureInfo", "WeakOutput", "ToolError", "default_classifier",
    "Policy", "default_ladders",
    "Reframe", "Escalate", "SwapTool", "Backoff", "Decompose",
    "respawn_chat", "ChatResult",
    "recover", "RecoveryResult", "uniform_attributor", "point_attributor",
]