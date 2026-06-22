"""Tests for respawn.llm using a fake anthropic-shaped client (no API key)."""
import json

from respawn import WeakOutput
from respawn.llm import respawn_chat


class _Block:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def create(self, model, messages, **kw):
        action = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return action(model, messages)


class FakeAnthropic:
    def __init__(self, script):
        self.messages = _Messages(script)


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


def test_rate_limit_then_success():
    def boom(m, x):
        raise RateLimitError("429 slow down")

    def ok(m, x):
        return _Resp("answer")

    res = respawn_chat(FakeAnthropic([boom, ok]),
                       messages=[{"role": "user", "content": "hi"}],
                       model="haiku", backoff_base=0.001)
    assert res.ok and res.text == "answer"


def test_bad_json_then_reframe():
    def chatty(m, x):
        return _Resp("here you go: 42")

    def clean(m, x):
        return _Resp('{"total": 42}')

    def must_be_json(text):
        try:
            json.loads(text)
        except Exception:
            raise WeakOutput("not valid JSON")

    res = respawn_chat(FakeAnthropic([chatty, clean]),
                       messages=[{"role": "user", "content": "json please"}],
                       model="haiku", validate=must_be_json)
    assert res.ok and json.loads(res.text) == {"total": 42}


def test_escalates_until_opus():
    def by_model(m, x):
        return _Resp(f"from {m}")

    def needs_opus(text):
        if "opus" not in text:
            raise WeakOutput("weak")

    res = respawn_chat(FakeAnthropic([by_model] * 5),
                       messages=[{"role": "user", "content": "hard"}],
                       model="haiku", escalate=["haiku", "sonnet", "opus"],
                       validate=needs_opus)
    assert res.ok and "opus" in res.text


def test_auth_error_fails_fast():
    def denied(m, x):
        raise AuthenticationError("401 bad key")

    res = respawn_chat(FakeAnthropic([denied] * 5),
                       messages=[{"role": "user", "content": "hi"}], model="haiku")
    assert not res.ok
    assert res.receipt.attempts == 1   # FATAL -> no wasted retries