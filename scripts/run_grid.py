"""Run the Phase 5 grid on Claude: every (method, question) listed under `grid` in configs/experiment.yaml.

  uv run python scripts/run_grid.py --dry-run   # cost estimate only, nothing is sent
  uv run python scripts/run_grid.py             # run; finished runs are skipped, so it resumes after a crash

Every paid call goes through the $7 Budget, which stops the grid before the limit is crossed.
"""
import argparse
import json
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from rlm_logistics import methods
from rlm_logistics.budget import Budget, BudgetExceeded
from rlm_logistics.llm import LONG_PROMPT, LLM

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAG_PROMPT, RLM_ROOT = 3_000, 20_000  # typical input tokens of one RAG call / of an RLM's root turns


def plan(qs: pd.DataFrame, grid: list[dict]):
    """(method, question) pairs in grid order."""
    for spec in grid:
        sel = qs[qs["size"].isin(spec["sizes"])]
        if "tiers" in spec:
            sel = sel[sel.tier.isin(spec["tiers"])]
        if "per_tier" in spec:
            sel = sel.groupby(["size", "tier"]).head(spec["per_tier"])
        yield from ((spec["method"], q) for q in sel.to_dict("records"))


def estimate(method: str, q: dict, ctx_tokens: int, llm: LLM) -> float:
    """Rough upper estimate in $. RLM sub-calls may read the whole context once for T2/T3."""
    if method == "direct_llm":
        tokens = ctx_tokens
    elif method.startswith("rag"):
        tokens = RAG_PROMPT
    else:
        reads = 0 if method == "rlm_depth0" or q["tier"] == "T1" else ctx_tokens
        tokens = RLM_ROOT + reads * (1.3 if method == "rlm_depth2" else 1.1)
    p_in, p_out = llm.long_price if method == "direct_llm" and tokens > LONG_PROMPT else llm.price
    return (tokens * p_in + 5_000 * p_out) / 1e6


def main(dry_run: bool):
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    meta = json.loads((PROCESSED / "contexts.json").read_text())
    qs = pd.read_json(PROCESSED / "questions.jsonl", lines=True)
    budget = Budget()
    llm = LLM.from_config("final", budget=budget)

    done = {m: methods.logged(m) for m in {s["method"] for s in cfg["grid"]}}
    todo = [(m, q) for m, q in plan(qs, cfg["grid"]) if q["id"] not in done[m]]
    est = pd.DataFrame([{"method": m, "cost": estimate(m, q, meta[q["size"]]["tokens"], llm)} for m, q in todo])
    if est.empty:
        print("Nothing left to run.")
        return
    print(est.groupby("method", sort=False).cost.agg(["count", "sum"]).round(3).rename(columns={"sum": "est_$"}))
    left = budget.limit - budget.spent
    print(f"\n{len(todo)} runs, estimated ${est.cost.sum():.2f}; budget left ${left:.2f} of ${budget.limit:.2f}")
    if dry_run:
        return

    lines = (PROCESSED / "context_2m.txt").read_text(encoding="utf-8").split("\n")
    context = cache(lambda size: "\n".join(lines[:meta[size]["n_records"]]))

    @cache
    def retriever(name):
        from rlm_logistics.rag import BM25, Dense, Hybrid
        bm25 = BM25(lines)
        return bm25 if name == "bm25" else Hybrid(bm25, Dense(lines, PROCESSED / "embeddings_e5small.npy"))

    def ask(method: str, q: dict):
        n = meta[q["size"]]["n_records"]
        if method == "direct_llm":
            return methods.direct(q["question"], llm, context(q["size"]))
        if method.startswith("rag_"):
            return methods.rag(q["question"], llm, lines, retriever(method[4:]), n)
        return methods.rlm(q["question"], llm, context(q["size"]), depth=int(method[-1]))

    for i, (method, q) in enumerate(todo, 1):
        try:
            row = methods.run(method, q, lambda: ask(method, q), llm.model)
        except BudgetExceeded as e:
            print(f"Stopped: {e}")
            break
        except Exception as e:  # an API or code error loses one run, not the grid; it is retried next time
            print(f"[{i}/{len(todo)}] {method} {q['id']}: ERROR {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"[{i}/{len(todo)}] {method:<11} {q['id']:<12} score={row['score']:.2f} "
              f"${row['cost']:.4f} {row['seconds']:.0f}s | total ${budget.spent:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args().dry_run)
