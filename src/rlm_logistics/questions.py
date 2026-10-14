"""Benchmark questions and their exact answers, computed with pandas from the review labels.

Each context is a DataFrame of the orders it contains, with a `label` column.
T1 lookup (one record), T2 aggregation (all records), T3 pairwise (pairs of records).
"""
from itertools import chain, zip_longest

import pandas as pd

NEGATIVE = ["delivery", "product", "seller_service"]
KIND = {
    "delivery": "a delivery complaint (late, lost, not received, damaged in transit)",
    "product": "a product complaint (quality, defect, not as described)",
    "seller_service": "a seller-service complaint (communication, refund, wrong or missing items)",
    "positive": "positive (no real complaint)",
    "negative": "any kind of complaint",
    "not_delivery": "anything other than a delivery complaint",
}
GROUP = {"negative": NEGATIVE, "not_delivery": ["product", "seller_service", "positive", "other"]}
LABEL_HINT = "Answer with exactly one of: delivery, product, seller_service, positive, other."
PAIR_HINT = "Answer as a list of order-id pairs, e.g. [[id1, id2], [id3, id4]]."

# (scope with {v}, filter column, pairing key, kind a, kind b, late orders only)
T3_SPECS = [
    ("orders whose customer is in state {v}", "customer_state", "customer_city", "delivery", "product", False),
    ("orders whose seller is in state {v}", "seller_state", "sid", "positive", "negative", False),
    ("late orders whose customer is in state {v}", "customer_state", "category", "delivery", "not_delivery", True),
]
PAIR_KEY = {"customer_city": "the same customer city", "sid": "the same seller", "category": "the same product category"}


def t1(ctx: pd.DataFrame, n: int, seed: int) -> list:
    """Needle in a haystack: the complaint of one specific order."""
    picks = ctx[ctx.label.isin(NEGATIVE)].sample(n, random_state=seed)
    return [(f"What is the main complaint in the review of order {r.oid}? {LABEL_HINT}", r.label, "label")
            for r in picks.itertuples()]


def t2(ctx: pd.DataFrame, n: int) -> list:
    """A percentage that needs the label of every matching record."""
    state, cat = ctx.customer_state.mode()[0], ctx.category.mode()[0]
    specs = [
        ("orders delivered after the estimated date", ctx.days_late > 0, "delivery"),
        ("orders delivered on or before the estimated date", ctx.days_late <= 0, "product"),
        ("orders with a review score of 1 or 2", ctx.review_score <= 2, "seller_service"),
        (f"orders whose customer is in state {state}", ctx.customer_state == state, "positive"),
        (f"orders in the product category {cat}", ctx.category == cat, "product"),
        ("orders where the seller and customer are in different states", ctx.seller_state != ctx.customer_state, "delivery"),
    ]
    return [(f"Among {desc}, what percentage of reviews count as {KIND[label]}? Answer with a number rounded to one decimal.",
             round(100 * (ctx[mask].label == label).mean(), 1), "number")
            for desc, mask, label in specs[:n]]


def pairs(sub: pd.DataFrame, key: str, a: str, b: str) -> list[list[str]]:
    """All order pairs sharing `key`, where one order is of kind a and the other of kind b."""
    m = sub[sub.label == a].merge(sub[sub.label.isin(GROUP.get(b, [b]))], on=key)
    return sorted(sorted(p) for p in zip(m.oid_x, m.oid_y))


def t3(ctx: pd.DataFrame, n: int, lo: int = 3, hi: int = 60) -> list:
    """Pairwise questions. The scope (all orders, or one state) is chosen so the answer has lo..hi pairs."""
    per_spec = []
    for scope, col, key, a, b, late_only in T3_SPECS:
        base = ctx[ctx.days_late > 0] if late_only else ctx
        options = [("all " + scope.split(" whose")[0], base)]
        options += [(scope.format(v=v), base[base[col] == v]) for v in sorted(base[col].unique())]
        per_spec.append([(f"Among {s}, list every pair of orders from {PAIR_KEY[key]} where one review is {KIND[a]} "
                          f"and the other is {KIND[b]}. {PAIR_HINT}", ans, "pairs")
                         for s, sub in options if lo <= len(ans := pairs(sub, key, a, b)) <= hi])
    return [q for q in chain(*zip_longest(*per_spec)) if q][:n]  # round-robin across the 3 specs


def build_questions(contexts: dict[str, pd.DataFrame], n: dict[str, int], seed: int = 42) -> pd.DataFrame:
    """contexts: {size_name: labeled orders}; n: {size_name: questions per tier}."""
    rows = []
    for name, ctx in contexts.items():
        for tier, qs in [("T1", t1(ctx, n[name], seed)), ("T2", t2(ctx, n[name])), ("T3", t3(ctx, n[name]))]:
            rows += [{"id": f"{name}-{tier}-{i}", "size": name, "tier": tier, "question": qu, "answer": ans,
                      "answer_type": kind} for i, (qu, ans, kind) in enumerate(qs)]
    return pd.DataFrame(rows)
