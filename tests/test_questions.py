import pandas as pd

from rlm_logistics.data import build_contexts, pack
from rlm_logistics.questions import pairs, t2


def test_pack_finds_largest_prefix():
    lines = ["aaaa"] * 10            # each line 4 chars; joined with "\n"
    assert pack(lines, 14, len) == 3  # "aaaa\naaaa\naaaa" = 14 chars
    assert build_contexts(lines, [14, 24], len) == {14: 3, 24: 5}


def test_pairs_match_on_key_and_labels():
    sub = pd.DataFrame({"oid": ["a", "b", "c", "d"], "city": ["x", "x", "x", "y"],
                        "label": ["delivery", "product", "product", "product"]})
    assert pairs(sub, "city", "delivery", "product") == [["a", "b"], ["a", "c"]]


def test_t2_percentage():
    ctx = pd.DataFrame({"days_late": [3, 5, -1, 0], "review_score": [1, 2, 5, 4], "category": ["c"] * 4,
                        "customer_state": ["SP"] * 4, "seller_state": ["SP"] * 4,
                        "label": ["delivery", "product", "positive", "product"]})
    question, answer, kind = t2(ctx, 1)[0]
    assert "after the estimated date" in question and answer == 50.0 and kind == "number"


def test_parse_compact_codes_skips_bad_numbers():
    from rlm_logistics.labels import parse
    out = parse('{"codes": "0g 1d 1p 7s 2 o"}', ["a", "b", "c"], translate=False)
    assert out == [{"order_id": "a", "label": "positive"}, {"order_id": "b", "label": "product"},
                   {"order_id": "c", "label": "other"}]
