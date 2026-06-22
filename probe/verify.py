"""Answer verification for the re-execution probe.

Who&When ground-truth answers are short (numbers, short phrases). We normalize
both sides and accept an exact normalized match or a clean containment. This is
deliberately simple and its limits are stated in PROBE.md: a stricter verifier
(or an LLM judge) would change absolute recovery numbers but not the *comparison*
between re-entry points, which is what the probe measures.
"""
from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}
_FINAL = re.compile(r"final answer\s*[:\-]?\s*(.+)", re.I | re.S)


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def extract_final(text: str) -> str:
    """Pull the answer after a 'FINAL ANSWER:' marker if present, else use all text."""
    m = _FINAL.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def is_correct(prediction: str, ground_truth: str) -> bool:
    pred = normalize(extract_final(prediction))
    truth = normalize(ground_truth)
    if not truth:
        return False
    if pred == truth:
        return True
    # short ground truths: accept exact token/phrase containment
    return bool(truth) and (f" {truth} " in f" {pred} " or pred == truth)