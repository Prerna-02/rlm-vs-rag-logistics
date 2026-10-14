"""Scoring: pull the final answer out of a model's reply and compare it with the answer key.

T1 label  -> 1 if the label matches, else 0
T2 number -> 1 if within TOLERANCE percentage points, else 0
T3 pairs  -> F1 between the predicted and true sets of order pairs (order inside a pair ignored)
"""
import re

from .labels import LABELS

TOLERANCE = 2.0
PAIR = re.compile(r"\[\s*[\"']?(O\d{5})[\"']?\s*,\s*[\"']?(O\d{5})[\"']?\s*\]")


def final_answer(text: str) -> str:
    """Text after the last 'ANSWER:' (models are asked to end with it); the whole reply otherwise."""
    return text.rsplit("ANSWER:", 1)[-1]


def f1(pred: set, true: set) -> float:
    hits = len(pred & true)
    return 2 * hits / (len(pred) + len(true)) if pred or true else 1.0


def score(answer_type: str, reply: str, truth) -> float:
    ans = final_answer(reply)
    if answer_type == "label":
        found = re.findall("|".join(LABELS), ans.lower())
        return float(bool(found) and found[0] == truth)
    if answer_type == "number":
        m = re.search(r"-?\d+(?:\.\d+)?", ans)
        return float(bool(m) and abs(float(m.group()) - truth) <= TOLERANCE)
    return f1({frozenset(p) for p in PAIR.findall(ans)}, {frozenset(p) for p in truth})
