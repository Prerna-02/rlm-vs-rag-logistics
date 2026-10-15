"""Export the experiment results as small JSON files for the static demo site in demo/.

  uv run python scripts/build_demo.py

Writes demo/data/results.json (leaderboard, per-size scores, every run) and one trimmed
trajectory file per RLM run under demo/data/traces/. Only research data is exported:
per-question cost and time, never budgets or spending totals.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "demo" / "data"
METHODS = ["rag_bm25", "rag_hybrid", "direct_llm", "rlm_depth0", "rlm_depth1", "rlm_depth2"]
SIZES = ["32k", "128k", "512k", "2m"]
TIERS = ["T1", "T2", "T3"]


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"\n… [{len(text) - n:,} more characters]"


def main():
    runs = pd.concat(pd.read_json(p, lines=True) for p in RUNS.glob("*.jsonl"))
    qs = pd.read_json(ROOT / "data" / "processed" / "questions.jsonl", lines=True)
    meta = json.loads((ROOT / "data" / "processed" / "contexts.json").read_text())

    cell = runs.groupby(["method", "tier", "size"]).score.agg(["mean", "size"])
    by_size = {m: {t: {s: ({"score": round(cell.loc[(m, t, s), "mean"], 3), "n": int(cell.loc[(m, t, s), "size"])}
                           if (m, t, s) in cell.index else None) for s in SIZES} for t in TIERS} for m in METHODS}
    hard = runs[runs.tier != "T1"]
    leaderboard = [{
        "method": m,
        **{t: round(runs[(runs.method == m) & (runs.tier == t)].score.mean(), 3)
           if ((runs.method == m) & (runs.tier == t)).any() else None for t in TIERS},
        "hard": round(hard[hard.method == m].score.mean(), 3),
        "cost_median": round(runs[runs.method == m].cost.median(), 5),
        "cost_p90": round(runs[runs.method == m].cost.quantile(0.9), 5),
        "seconds_median": round(runs[runs.method == m].seconds.median(), 1),
        "runs": int((runs.method == m).sum()),
    } for m in METHODS]

    run_rows = [{
        "id": r.id, "method": r.method, "size": r.size, "tier": r.tier, "score": round(r.score, 3),
        "cost": round(r.cost, 5), "seconds": r.seconds, "answer": clip(r.reply.removeprefix("ANSWER:").strip(), 400),
        "steps": None if pd.isna(r.steps) else int(r.steps),
        "sub_calls": None if pd.isna(r.sub_calls) else int(r.sub_calls),
    } for r in runs.itertuples()]
    questions = {q.id: {"question": q.question, "type": q.answer_type,
                        "truth": q.answer if q.answer_type != "pairs" else f"{len(q.answer)} pairs",
                        "truth_pairs": q.answer if q.answer_type == "pairs" else None} for q in qs.itertuples()}

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "traces").mkdir(parents=True)
    for path in (RUNS / "trajectories").glob("*/*.json"):
        steps = [{"response": clip(s["response"], 4000), "outputs": [clip(o, 2000) for o in s["outputs"]],
                  "sub_calls": [{"prompt_chars": c["prompt_chars"], "reply": clip(c["reply"], 300)} for c in s["sub_calls"]]}
                 for s in json.loads(path.read_text(encoding="utf-8"))]
        dest = OUT / "traces" / path.parent.name / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(steps, ensure_ascii=False), encoding="utf-8")

    data = {"methods": METHODS, "sizes": SIZES, "tiers": TIERS, "model": "claude-haiku-5-5",
            "contexts": {s: {"records": meta[s]["n_records"], "tokens": meta[s]["tokens"]} for s in SIZES},
            "leaderboard": leaderboard, "by_size": by_size, "runs": run_rows, "questions": questions}
    (OUT / "results.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{len(run_rows)} runs, {sum(1 for _ in (OUT / 'traces').glob('*/*.json'))} trajectories -> {OUT}")


if __name__ == "__main__":
    main()
