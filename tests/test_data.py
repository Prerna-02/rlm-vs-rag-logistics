import pandas as pd

from rlm_logistics.data import serialize


def row(days_late, comment="chegou atrasado"):
    return pd.Series({
        "order_id": "o1", "seller_id": "s1", "seller_state": "SP", "seller_city": "sao paulo",
        "customer_state": "RJ", "customer_city": "rio de janeiro", "category": "bed_bath_table",
        "order_purchase_timestamp": pd.Timestamp("2017-11-02"),
        "order_estimated_delivery_date": pd.Timestamp("2017-11-20"),
        "order_delivered_customer_date": pd.Timestamp("2017-11-27"),
        "days_late": days_late, "review_score": 1.0, "comment": comment,
    })


def test_serialize_late_with_comment():
    s = serialize(row(7))
    assert s.startswith("[order o1] seller=s1 (SP/sao paulo) customer=RJ/rio de janeiro")
    assert '(7 days late) review=1/5 "chegou atrasado"' in s


def test_serialize_status_and_missing_review():
    assert "(on time)" in serialize(row(0))
    assert "(3 days early) review=none" in serialize(row(-3, comment=None))
