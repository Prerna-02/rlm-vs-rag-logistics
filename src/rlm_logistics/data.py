"""Load the Olist CSVs, join them into one row per order, and turn each row into a text record."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

TABLES = {
    "orders": "olist_orders_dataset",
    "reviews": "olist_order_reviews_dataset",
    "items": "olist_order_items_dataset",
    "sellers": "olist_sellers_dataset",
    "customers": "olist_customers_dataset",
    "products": "olist_products_dataset",
    "categories": "product_category_name_translation",
}
DATES = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]


def load_tables(raw: Path = RAW) -> dict[str, pd.DataFrame]:
    """Read the 7 CSVs we use into a dict of DataFrames, keyed by short name."""
    return {name: pd.read_csv(raw / f"{file}.csv") for name, file in TABLES.items()}


def build_orders(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per delivered, single-seller order, with locations, category, delay and its latest review."""
    items = t["items"].merge(t["products"], on="product_id").merge(t["categories"], on="product_category_name", how="left")
    single_seller = items.groupby("order_id")["seller_id"].transform("nunique") == 1
    items = (items[single_seller]
             .groupby("order_id")
             .agg(seller_id=("seller_id", "first"), category=("product_category_name_english", "first"),
                  n_items=("order_item_id", "count"), freight=("freight_value", "sum"))
             .reset_index())

    reviews = (t["reviews"].sort_values("review_answer_timestamp")
               .drop_duplicates("order_id", keep="last")
               [["order_id", "review_score", "review_comment_message"]]
               .rename(columns={"review_comment_message": "comment"}))

    df = (t["orders"].query("order_status == 'delivered'")
          .merge(items, on="order_id")
          .merge(t["sellers"], on="seller_id")
          .merge(t["customers"], on="customer_id")
          .merge(reviews, on="order_id", how="left"))
    df[DATES] = df[DATES].apply(pd.to_datetime)
    df = df.dropna(subset=["order_delivered_customer_date"])
    df["days_late"] = (df["order_delivered_customer_date"].dt.normalize() - df["order_estimated_delivery_date"]).dt.days
    df["category"] = df["category"].fillna("unknown")
    df["comment"] = df["comment"].str.replace(r"\s+", " ", regex=True).str.strip()
    # Short ids (O01234, S0567): the 32-character hex ids cost ~20 tokens each; these cost ~3.
    # Category codes follow the sorted original ids, so the mapping is the same on every run.
    df["oid"] = "O" + df.order_id.astype("category").cat.codes.astype(str).str.zfill(5)
    df["sid"] = "S" + df.seller_id.astype("category").cat.codes.astype(str).str.zfill(4)
    return df


def claude_token_counter(model: str, max_chars: int = 1_500_000):
    """Return count(text) using Claude's own tokenizer; long texts are counted in pieces and summed."""
    import anthropic

    client = anthropic.Anthropic()

    def count(text: str) -> int:
        pieces, cur = [], ""
        for line in text.split("\n"):
            if cur and len(cur) + len(line) > max_chars:
                pieces.append(cur)
                cur = ""
            cur += line + "\n"
        pieces.append(cur)
        return sum(client.messages.count_tokens(model=model, messages=[{"role": "user", "content": p}]).input_tokens
                   for p in pieces)
    return count


def pack(lines: list[str], target: int, count, lo: int = 0) -> int:
    """Binary search: the largest n such that the first n lines fit within `target` tokens."""
    hi = len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count("\n".join(lines[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def build_contexts(lines: list[str], sizes: list[int], count) -> dict[int, int]:
    """Nested contexts: each size is a prefix of the shuffled records. Returns {size: n_records}."""
    n, out = 0, {}
    for size in sorted(sizes):
        out[size] = n = pack(lines, size, count, lo=n)
    return out


def serialize(row) -> str:
    """Turn one order row into the single-line text record the LLMs will read."""
    late = row.days_late
    status = f"{late} days late" if late > 0 else "on time" if late == 0 else f"{-late} days early"
    review = f'review={row.review_score:.0f}/5 "{row.comment}"' if pd.notna(row.comment) else "review=none"
    return (f"[order {row.oid}] seller={row.sid} ({row.seller_state}/{row.seller_city}) "
            f"customer={row.customer_state}/{row.customer_city} category={row.category} "
            f"purchased={row.order_purchase_timestamp:%Y-%m-%d} estimated={row.order_estimated_delivery_date:%Y-%m-%d} "
            f"delivered={row.order_delivered_customer_date:%Y-%m-%d} ({status}) {review}")
