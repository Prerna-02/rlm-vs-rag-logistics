"""The methods under test. Each takes a question and returns a Reply; run() scores and logs it.

Every method gets the same system prompt (record format, label definitions, answer format),
so differences in score come from how much of the context each method gets to see.
"""
import json
from collections.abc import Callable
from pathlib import Path

from .eval import score
from .labels import LABEL_RULES
from .llm import LLM, Reply

RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"

SYSTEM = f"""You answer questions about Brazilian e-commerce orders. Each record is one line:
[order ID] seller=ID (state/city) customer=state/city category=... purchased/estimated/delivered dates (delivery status) review=score/5 "comment in Portuguese"
Questions about complaints use these review labels:
{LABEL_RULES}
Use only the records you are given. Think step by step if needed, then end with one final line:
ANSWER: <answer>"""


def direct(question: str, llm: LLM, context: str) -> Reply:
    """Direct LLM: the whole context in one prompt."""
    return llm.chat(f"Records:\n{context}\n\nQuestion: {question}", SYSTEM)


def rag(question: str, llm: LLM, lines: list[str], retriever, n: int, k: int = 20) -> Reply:
    """RAG: only the k records the retriever ranks highest (among the first n, i.e. this context)."""
    hits = "\n".join(lines[i] for i in retriever.search(question, k, n))
    return llm.chat(f"Records (the {k} most relevant out of {n}):\n{hits}\n\nQuestion: {question}", SYSTEM)


def logged(method: str, out: Path = RUNS) -> dict[str, dict]:
    """Runs already saved for this method, by question id (lets an interrupted grid resume)."""
    path = out / f"{method}.jsonl"
    if not path.exists():
        return {}
    return {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def run(method: str, q: dict, ask: Callable[[], Reply], model: str, out: Path = RUNS) -> dict:
    """Score one reply and append it to <out>/<method>.jsonl. `ask` is only called if this
    question has no saved run yet, so re-running a notebook or grid never pays twice."""
    if (row := logged(method, out).get(q["id"])):
        return row
    reply = ask()
    row = {"id": q["id"], "size": q["size"], "tier": q["tier"], "method": method, "model": model,
           "score": score(q["answer_type"], reply.text, q["answer"]), "reply": reply.text,
           "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
           "cost": reply.cost, "seconds": round(reply.seconds, 1)}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{method}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
