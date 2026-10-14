# RLM vs RAG on Logistics Data

Benchmarking **Recursive Language Models (RLM)** against **RAG** and **direct LLM calls** on real e-commerce delivery data, to find out when RLM wins and what it costs.

> **Scope:** the data is Brazilian e-commerce *last-mile delivery* (Olist), not freight or port logistics.

**Status:** Phase 1 (setup + data exploration) of 6.

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and (for development runs) [Ollama](https://ollama.com) with `qwen3:8b`.

```bash
git clone https://github.com/Prerna-02/rlm-vs-rag-logistics.git
cd rlm-vs-rag-logistics
uv sync
cp .env.example .env          # then add your KAGGLE_API_TOKEN
uv run python scripts/download_data.py
uv run pytest
```

Open `notebooks/` in VS Code and pick the `.venv` kernel.

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 00 | `00_concepts` | Context rot, RAG and RLM explained with toy examples |
| 01 | `01_data_exploration` | Loading and joining Olist, delays, reviews, context-size estimates |

## Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0). Raw files are not committed; the download script fetches them.

## References

- Zhang, Kraska & Khattab, *Recursive Language Models* (MIT): [arXiv](https://arxiv.org/abs/2512.24601) · [code](https://github.com/alexzhang13/rlm)
