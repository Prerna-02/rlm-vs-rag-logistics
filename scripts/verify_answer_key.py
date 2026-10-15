"""Rebuild the benchmark's answer key from the raw Olist data and check it matches the committed one.

  uv run python scripts/download_data.py      # needs KAGGLE_API_TOKEN in .env
  uv run python scripts/verify_answer_key.py

No paid API calls: the records are rebuilt and shuffled exactly as in notebook 02, joined with the
committed labels, and every question and answer is recomputed with pandas.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from rlm_logistics.data import build_orders, load_tables
from rlm_logistics.questions import build_questions

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

if __name__ == "__main__":
    meta = json.loads((PROCESSED / "contexts.json").read_text())
    records = (build_orders(load_tables()).dropna(subset=["comment"])
               .sample(frac=1, random_state=42).reset_index(drop=True))
    records = records.merge(pd.read_csv(ROOT / "data" / "labels" / "labels.csv"), on="order_id", how="left")
    contexts = {name: records.iloc[:m["n_records"]] for name, m in meta.items()}
    rebuilt = build_questions(contexts, {"32k": 6, "128k": 6, "512k": 6, "2m": 2})

    committed = pd.read_json(PROCESSED / "questions.jsonl", lines=True)
    cols = ["id", "question", "answer", "n"]
    same = rebuilt[cols].astype(str).equals(committed[cols].astype(str))
    print(f"{len(rebuilt)} questions rebuilt; identical to data/processed/questions.jsonl: {same}")
    sys.exit(0 if same else 1)
