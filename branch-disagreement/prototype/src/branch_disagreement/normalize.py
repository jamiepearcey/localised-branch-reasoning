"""Deterministic short-answer normalization and correctness matching.

SQuAD-style normalization: lowercase, strip punctuation and articles, collapse
whitespace. Correctness is exact match of the normalized prediction against any
normalized gold answer (or gold contained as a whole-token span). This is
deliberately simple and frozen — see ``.context/invariants.md``.
"""

import re
import string
from typing import Iterable, List

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation and articles, collapse whitespace."""
    if text is None:
        return ""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def normalized_tokens(text: str) -> List[str]:
    norm = normalize_answer(text)
    return norm.split() if norm else []


def answer_matches(prediction: str, gold_answers: Iterable[str]) -> bool:
    """True if the prediction matches any gold answer.

    Matches when the normalized prediction equals a normalized gold answer, or
    when a (multi-word) gold answer appears as a contiguous token span inside the
    prediction. The span rule lets a longer generated phrase still count when it
    contains the exact gold answer.
    """
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    pred_tokens = pred_norm.split()
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if not gold_norm:
            continue
        if pred_norm == gold_norm:
            return True
        gold_tokens = gold_norm.split()
        if _contains_span(pred_tokens, gold_tokens):
            return True
    return False


def _contains_span(haystack: List[str], needle: List[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return True
    return False
