"""Offline demo of respawn.llm -- no API key needed.

A fake, anthropic-shaped client raises real-looking SDK exceptions so you can
watch respawn diagnose each one and retry differently. Swap in a real
`anthropic.Anthropic()` or `openai.OpenAI()` and the same code works.

    python demo_llm.py
"""
import json

from respawn import WeakOutput
from respawn.llm import respawn_chat


# --- minimal fakes that look like the anthropic SDK -------------------------
class RateLimitError(Exception):
    status_code = 429
class APITimeoutError(Exception):
    pass
class AuthenticationError(Exception):
    status_code = 401

class _Block:
    def __init__(self, text): self.text = text
class _Resp:
    def __init__(self, text): self.content = [_Block(text)]

class FakeMessages:
    def __init__(self, script): self._script = list(script); self._i = 0
    def create(self, model, messages, **kw):
        action = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return action(model, messages)

class FakeAnthropic:
    """type(self).__module__ won't contain 'anthropic', so duck-typing kicks in."""
    def __init__(self, script): self.messages = FakeMessages(script)


# --- Case 1: rate limit, then a clean response ------------------------------
def case_rate_limit():
    def boom(m, msgs): raise RateLimitError("429: slow down")
    def ok(m, msgs):   return _Resp("the answer")
    client = FakeAnthropic([boom, ok])
    return respawn_chat(client, messages=[{"role": "user", "content": "hi"}],
                        model="haiku", backoff_base=0.01)


# --- Case 2: model returns non-JSON until reframed --------------------------
def case_bad_json():
    def chatty(m, msgs): return _Resp("Sure! Here you go: total is 42 dollars.")
    def clean(m, msgs):  return _Resp('{"total": 42}')
    client = FakeAnthropic([chatty, clean])

    def must_be_json(text):
        try: json.loads(text)
        except Exception: raise WeakOutput("response was not valid JSON")

    return respawn_chat(client, messages=[{"role": "user", "content": "give JSON"}],
                        model="haiku", validate=must_be_json)


# --- Case 3: weak answer until we escalate up the ladder --------------------
def case_escalate():
    def by_model(m, msgs): return _Resp(f"answer from {m}")
    client = FakeAnthropic([by_model] * 5)

    def needs_opus(text):
        if "opus" not in text: raise WeakOutput(f"weak answer: {text!r}")

    return respawn_chat(
        client, messages=[{"role": "user", "content": "hard question"}],
        model="haiku", escalate=["haiku", "sonnet", "opus"], validate=needs_opus)


# --- Case 4: bad API key -> FATAL, fail fast (no wasted retries) -------------
def case_fatal():
    def denied(m, msgs): raise AuthenticationError("401: invalid x-api-key")
    client = FakeAnthropic([denied] * 5)
    return respawn_chat(client, messages=[{"role": "user", "content": "hi"}],
                        model="haiku")


if __name__ == "__main__":
    for title, fn in [
        ("rate limit -> back off", case_rate_limit),
        ("non-JSON -> reframe with the error", case_bad_json),
        ("weak answer -> escalate haiku->sonnet->opus", case_escalate),
        ("bad API key -> FATAL, fail fast", case_fatal),
    ]:
        res = fn()
        print(f"\n=== {title} ===")
        print(res.explain())
        print("text:", res.text)