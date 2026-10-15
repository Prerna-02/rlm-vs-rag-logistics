# RLM vs RAG on Logistics Data

Benchmarking **Recursive Language Models (RLM)** against **RAG** and **direct LLM calls** on real e-commerce delivery data, to find out when RLM wins and what it costs.

**Live results:** <https://rlm-vs-rag-logistics-demo.vercel.app> · **Report:** [`report/report.md`](report/report.md) ([PDF](report/report.pdf))

> **Scope:** the data is Brazilian e-commerce *last-mile delivery* (Olist), not freight or port logistics.

## Headline results

59 questions × 6 methods (297 runs) on Claude Haiku 5.5, over contexts of 32K, 128K, 512K and 2M tokens.

| Method | T1 lookup | T2 aggregation | T3 pairwise (F1) | Median cost / question |
|---|---|---|---|---|
| RAG, BM25 | **0.90** | 0.00 | 0.07 | $0.0006 |
| RAG, hybrid | 0.70 | 0.10 | 0.06 | $0.0006 |
| Direct LLM (≤ 512K) | 0.85 | 0.15 | 0.34 | $0.065 |
| RLM, depth 0 | 0.85 | 0.45 | 0.80 | $0.003 |
| **RLM, depth 1** | 0.85 | **0.60** | **0.84** | $0.003 |
| RLM, depth 2 (subset) | | 0.67 | 0.89 | $0.059 |

* **Finding one record:** RAG is as accurate as anything and the cheapest.
* **Counting or comparing across all records:** RAG and direct LLM collapse; RLM keeps most of its accuracy as the context grows.
* **One level of sub-calls is the sweet spot:** depth 2 barely improves on depth 1 at ~10× the median cost.
* Full analysis, failure taxonomy and limitations: [`notebooks/09_analysis.ipynb`](notebooks/09_analysis.ipynb) and the [report](report/report.md).

![Accuracy vs context size](report/figures/accuracy_vs_size.png)
![Cost vs accuracy](report/figures/cost_vs_accuracy.png)

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and (for development runs) [Ollama](https://ollama.com) with `qwen3:8b`.

```bash
git clone https://github.com/Prerna-02/rlm-vs-rag-logistics.git
cd rlm-vs-rag-logistics
uv sync
cp .env.example .env          # then fill in the keys and your spending cap (BUDGET_USD)
uv run python scripts/download_data.py
uv run pytest
```

Open `notebooks/` in VS Code and pick the `.venv` kernel. Notebook 07 needs Jupyter started with `PYTHONUTF8=1` on Windows.

## Reproducing the results

The answer key can be rebuilt from the raw data with no API calls:

```bash
uv run python scripts/verify_answer_key.py      # rebuilds all 59 questions and answers; checks they match
```

Re-running the paid steps:

```bash
uv run python scripts/label_reviews.py submit    # then `collect`: the answer key labels (Batch API)
uv run python scripts/run_grid.py --dry-run      # cost estimate for the experiment grid
uv run python scripts/run_grid.py                # every method x question; resumes after interruptions
```

Every paid call goes through a hard spending cap (`BUDGET_USD` in `.env`). Results land in `results/runs/` (one JSON line per run, plus RLM trajectories) and `results/metrics.csv`.
`uv run python scripts/build_demo.py` refreshes the site data in `demo/` (a static site, deployed on Vercel from `main`).

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 00 | `00_concepts` | Context rot, RAG and RLM explained with toy examples |
| 01 | `01_data_exploration` | Loading and joining Olist, delays, reviews, context-size estimates |
| 02 | `02_build_contexts` | Nested 32K-2M contexts measured with Claude's tokenizer |
| 03 | `03_benchmark_design` | Human audit of the labels, questions and exact answer key |
| 04 | `04_baseline_direct_llm` | Whole context in one prompt |
| 05 | `05_baseline_rag` | BM25, dense and hybrid retrieval; why RAG can't aggregate |
| 06 | `06_rlm_from_scratch` | Our RLM: REPL, sub-calls, depth 0/1/2, safety limits |
| 07 | `07_rlm_official_library` | The authors' `rlms` library on the same questions |
| 08 | `08_run_experiments` | The experiment grid and how it ran |
| 09 | `09_analysis` | Results for RQ1-RQ5, charts, failure taxonomy, limitations |

## Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0). Raw files are not committed; the download script fetches them.

## Limitations

One model and one run per question; 6 questions per cell (2 at 2M). The answer key is LLM-made and checked by a 200-review human audit (93.0% agreement after the labeling rules were sharpened). See notebook 09.

## References

- Zhang, Kraska & Khattab, *Recursive Language Models* (MIT): [arXiv](https://arxiv.org/abs/2512.24601) · [code](https://github.com/alexzhang13/rlm)
