"""Label the reviews used in the contexts with Claude (Batch API) and build the blind human-audit sheet.

Run notebook 02 first: it saves data/processed/records.csv (only the orders inside the contexts).

  uv run python scripts/label_reviews.py submit    # sends 2 batches, saves their ids
  uv run python scripts/label_reviews.py collect   # once both batches have ended (usually < 1 hour)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from rlm_logistics.labels import collect_batch, submit_batch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "labels"
STATE = OUT / "batches.json"
AUDIT_N, SEED = 200, 42

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    model = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))["labeler"]["model"]
    reviews = pd.read_csv(ROOT / "data" / "processed" / "records.csv")
    audit = reviews.sample(AUDIT_N, random_state=SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    if sys.argv[1] == "submit":
        ids = {"labels": submit_batch(reviews, model), "audit": submit_batch(audit, model, translate=True)}
        STATE.write_text(json.dumps(ids, indent=2))
        print(ids)
    else:
        ids = json.loads(STATE.read_text())
        labels = collect_batch(ids["labels"], reviews)
        labels.to_csv(OUT / "labels.csv", index=False)
        print(f"{len(labels):,} of {len(reviews):,} reviews labeled\n{labels.label.value_counts()}")
        # Blind sheet: the human sees the review and translation, NOT the LLM label.
        sheet = collect_batch(ids["audit"], audit, translate=True).merge(audit[["order_id", "review_score", "comment"]], on="order_id")
        sheet[["order_id", "review_score", "comment", "english"]].assign(human_label="").to_csv(
            OUT / "audit_sheet.csv", index=False, encoding="utf-8-sig")
        print(f"audit sheet: {len(sheet)} rows -> fill the human_label column")
