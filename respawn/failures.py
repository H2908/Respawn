"""Failure taxonomy and the default heuristic classifier.

A classifier maps (exception, attempt) -> FailureType. The default one uses
exception type + message heuristics; you can pass your own (e.g. an LLM judge)
to Respawn(classifier=...).
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Callable


class FailureType(enum.Enum):
    BAD_OUTPUT = "bad_output"     # validation failed / weak or malformed answer
    TOOL_ERROR = "tool_error"     # a tool/function raised
    RATE_LIMIT = "rate_limit"     # 429 / quota
    TIMEOUT = "timeout"           # slow / deadline exceeded
    LOOP = "loop"                 # repeating itself / no progress
    FATAL = "fatal"               # not worth retrying (auth, permission, bad key)
    UNKNOWN = "unknown"


@dataclass
class FailureInfo:
    type: FailureType
    error: BaseException
    message: str

    def __str__(self) -> str:  # what gets fed back into the next prompt
        return f"[{self.type.value}] {self.message}"


# Marker exception users can raise from a validator to signal a weak answer.
class WeakOutput(Exception):
    """Raise from your step/validator when the output is wrong-but-not-crashing."""


class ToolError(Exception):
    """Wrap a failing tool call so respawn can classify it as TOOL_ERROR."""


_RATE_LIMIT = re.compile(r"\b(429|rate.?limit|quota|too many requests)\b", re.I)
_TIMEOUT = re.compile(r"\b(timeout|timed out|deadline)\b", re.I)
_LOOP = re.compile(r"\b(loop|no progress|repeat)\b", re.I)


def default_classifier(error: BaseException, attempt: Attempt) -> FailureInfo:  # noqa: F821
    msg = str(error) or error.__class__.__name__
    if isinstance(error, WeakOutput):
        t = FailureType.BAD_OUTPUT
    elif isinstance(error, ToolError):
        t = FailureType.TOOL_ERROR
    elif isinstance(error, TimeoutError) or _TIMEOUT.search(msg):
        t = FailureType.TIMEOUT
    elif _RATE_LIMIT.search(msg):
        t = FailureType.RATE_LIMIT
    elif _LOOP.search(msg):
        t = FailureType.LOOP
    elif isinstance(error, (ValueError, KeyError, TypeError)):
        # malformed arg / hallucinated field land here most often
        t = FailureType.BAD_OUTPUT
    else:
        t = FailureType.UNKNOWN
    return FailureInfo(type=t, error=error, message=msg)


Classifier = Callable[[BaseException, "Attempt"], FailureInfo]  # noqa: F821