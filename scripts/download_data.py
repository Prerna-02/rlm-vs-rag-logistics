"""Download the Olist dataset from Kaggle into data/raw/.

Needs KAGGLE_API_TOKEN in .env (see .env.example).
Run:  uv run python scripts/download_data.py
"""
from pathlib import Path

from dotenv import load_dotenv

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

if __name__ == "__main__":
    load_dotenv()  # must run before importing kaggle: it reads the token on import
    import kaggle

    RAW.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files("olistbr/brazilian-ecommerce", path=RAW, unzip=True)
    print(*sorted(p.name for p in RAW.glob("*.csv")), sep="\n")
