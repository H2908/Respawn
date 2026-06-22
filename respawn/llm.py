"""respawn.llm -- wrap a raw LLM API call so a failed call retries *differently*.

Works with the anthropic and openai Python SDKs (auto-detected). One call:

    from respawn.llm import respawn_chat

    res = respawn_chat(
        client,                              # anthropic.Anthropic() or openai.OpenAI()
        messages=[{"role": "user", "content": "Extract the total as JSON"}],
        model="claude-haiku-4-5",            # starting rung
        escalate=["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"],
        validate=must_be_json,               # raise WeakOutput on bad content
        budget=4,
    )
    print(res.explain())   # what failed and how respawn recovered
    print(res.text)        # the final good output

respawn diagnoses the failure (rate limit, timeout, bad request, weak/invalid
output, auth) and retries with a changed strategy: back off, reframe with the
error fed back, or escalate up the model ladder. Auth/permission errors are
classified FATAL and fail fast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .core import Receipt, Respawn
from .failures import FailureInfo, FailureType, WeakOutput
from .policy import Policy
from .strategies import Backoff, Escalate, Reframe

Validator = Callable[[str], None]  # raise WeakOutput (or any Exception) if bad


# --------------------------------------------------------------------------- #
# Provider plumbing                                                           #
# --------------------------------------------------------------------------- #
def _provider(client: Any) -> str:
    mod = type(client).__module__.lower()
    if "anthropic" in mod:
        return "anthropic"
    if "openai" in mod:
        return "openai"
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        return "anthropic"
    if hasattr(client, "chat"):
        return "openai"
    raise ValueError("Unrecognized client; pass an anthropic or openai client.")


def _call(client: Any, provider: str, model: str, messages: list, **extra: Any):
    if provider == "anthropic":
        extra.setdefault("max_tokens", 1024)
        return client.messages.create(model=model, messages=messages, **extra)
    return client.chat.completions.create(model=model, messages=messages, **extra)


def _extract_text(provider: str, resp: Any) -> str:
    if provider == "anthropic":
        return "".join(getattr(b, "text", "") or "" for b in resp.content)
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# SDK-aware classifier                                                        #
# --------------------------------------------------------------------------- #
def llm_classifier(error: BaseException, attempt) -> FailureInfo:
    msg = str(error) or type(error).__name__
    if isinstance(error, WeakOutput):
        return FailureInfo(FailureType.BAD_OUTPUT, error, msg)

    names = " ".join(c.__name__.lower() for c in type(error).__mro__)
    status = getattr(error, "status_code", None)

    def has(*subs: str) -> bool:
        return any(s in names for s in subs)

    if has("ratelimit") or status == 429:
        t = FailureType.RATE_LIMIT
    elif has("authentication", "permission") or status in (401, 403):
        t = FailureType.FATAL
    elif has("timeout", "apiconnection"):
        t = FailureType.TIMEOUT
    elif has("badrequest", "invalidrequest", "unprocessable") or status in (400, 422):
        t = FailureType.BAD_OUTPUT
    elif has("validationerror") or isinstance(error, (ValueError, KeyError, TypeError)):
        t = FailureType.BAD_OUTPUT
    elif isinstance(status, int) and status >= 500:
        t = FailureType.TIMEOUT  # transient server error -> back off
    else:
        t = FailureType.UNKNOWN
    return FailureInfo(t, error, msg)


def build_llm_policy(escalate_models: List[str], backoff_base: float) -> Policy:
    esc = Escalate(ladder=escalate_models)
    back = Backoff(base=backoff_base)
    return Policy(ladders={
        FailureType.BAD_OUTPUT: [Reframe(), esc, esc],
        FailureType.RATE_LIMIT: [back, back, back],
        FailureType.TIMEOUT:    [back, esc],
        FailureType.LOOP:       [Reframe(), esc],
        FailureType.FATAL:      [],
        FailureType.UNKNOWN:    [back, Reframe()],
    })


# --------------------------------------------------------------------------- #
# Public entrypoint                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class ChatResult:
    ok: bool
    text: Optional[str]
    response: Any
    receipt: Receipt

    def explain(self) -> str:
        return self.receipt.explain()


def respawn_chat(
    client: Any,
    *,
    messages: list,
    model: str,
    escalate: Optional[List[str]] = None,
    validate: Optional[Validator] = None,
    budget: int = 4,
    backoff_base: float = 1.0,
    on_event: Optional[Callable[[str], None]] = None,
    **call_kwargs: Any,
) -> ChatResult:
    provider = _provider(client)
    escalate_models = escalate or [model]
    if model not in escalate_models:
        escalate_models = [model] + escalate_models
    policy = build_llm_policy(escalate_models, backoff_base)
    base_messages = list(messages)

    def step(a):
        corrections = a.meta.setdefault("corrections", [])
        if a.feedback:                       # Reframe wrote guidance -> add a turn
            corrections.append(a.feedback)
            a.feedback = ""
        msgs = base_messages + [
            {"role": "user", "content": c} for c in corrections
        ]
        resp = _call(client, provider, a.model, msgs, **call_kwargs)
        text = _extract_text(provider, resp)
        if validate is not None:
            validate(text)                   # raises -> classified as BAD_OUTPUT
        return resp

    r = Respawn(budget=budget, policy=policy,
                classifier=llm_classifier, on_event=on_event).run(step, model=model)

    text = _extract_text(provider, r.result) if r.success else None
    return ChatResult(ok=r.success, text=text, response=r.result, receipt=r)