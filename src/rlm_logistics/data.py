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
    return df


def serialize(row) -> str:
    """Turn one order row into the single-line text record the LLMs will read."""
    late = row.days_late
    status = f"{late} days late" if late > 0 else "on time" if late == 0 else f"{-late} days early"
    review = f'review={row.review_score:.0f}/5 "{row.comment}"' if pd.notna(row.comment) else "review=none"
    return (f"[order {row.order_id}] seller={row.seller_id} ({row.seller_state}/{row.seller_city}) "
            f"customer={row.customer_state}/{row.customer_city} category={row.category} "
            f"purchased={row.order_purchase_timestamp:%Y-%m-%d} estimated={row.order_estimated_delivery_date:%Y-%m-%d} "
            f"delivered={row.order_delivered_customer_date:%Y-%m-%d} ({status}) {review}")
