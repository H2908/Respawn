from __future__ import annotations

from enum import Enum


class Strategy(str, Enum):
    TRUNCATE_AND_REPLAY = "truncate_and_replay"
    PATCH_AND_CONTINUE = "patch_and_continue"
    ESCALATE = "escalate"
    ABORT = "abort"
