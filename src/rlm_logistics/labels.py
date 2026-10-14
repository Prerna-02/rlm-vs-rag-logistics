"""Label each review with one complaint category (the ground truth for the answer key).

Final labels come from Claude Haiku via the Batch API (half price). The same prompt can run on
local Ollama to test the pipeline for free. With translate=True the model also returns an English
translation, used for the human audit sheet.
"""
import json
import re
import urllib.request

import pandas as pd

LABELS = ["delivery", "product", "seller_service", "positive", "other"]
CODES = {"d": "delivery", "p": "product", "s": "seller_service", "g": "positive", "o": "other"}
CHUNK = 50  # reviews per request

SYSTEM = """You classify Brazilian e-commerce customer reviews (Portuguese) by their MAIN COMPLAINT.
Labels (code: meaning):
- d (delivery): complaint about late, slow, lost or not-received delivery, or damage in transit
- p (product): complaint about quality, defects, wrong size/colour, not as described, product broke
- s (seller_service): complaint about the seller: communication, refund, cancellation, wrong or missing items sent, invoice
- g (positive): praise or satisfaction with no real complaint, even if it mentions fast delivery or a good product
- o (other): anything else, or too vague to tell
Rules:
1. Never arrived / not received -> d, even if the seller also did not reply.
2. Wrong item, missing items or wrong quantity sent -> s.
3. Arrived broken or defective -> p.
4. Neutral but satisfied ("ok", "as expected", "received") -> g.
5. Complaints about price or shipping cost -> o.
6. Several complaints (e.g. late AND another problem): choose the one the customer stresses most."""


def schema(translate: bool) -> dict:
    """Labels only: one compact string such as "0g 1d 2p" (~3 output tokens per review).
    With translation (audit sample): one object per review with its id, label and English text."""
    if not translate:
        return {"type": "object", "properties": {"codes": {"type": "string"}},
                "required": ["codes"], "additionalProperties": False}
    item = {"id": {"type": "integer"}, "label": {"type": "string", "enum": LABELS}, "english": {"type": "string"}}
    results = {"type": "object", "properties": item, "required": list(item), "additionalProperties": False}
    return {"type": "object", "properties": {"results": {"type": "array", "items": results}},
            "required": ["results"], "additionalProperties": False}


def prompt(comments: list[str], translate: bool) -> str:
    task = ("For each review, give its full label name and an English translation." if translate else
            'Reply with the review number followed by its code, for every review, separated by spaces: "0g 1d 2p ..."')
    return task + "\n\n" + "\n".join(f"{i}: {c}" for i, c in enumerate(comments))


def parse(text: str, order_ids: list[str], translate: bool) -> list[dict]:
    """Map the model's numbered results back to order ids; numbers it skipped or invented are dropped."""
    data = json.loads(text)
    if translate:
        rows = [(r.pop("id"), r) for r in data["results"]]
    else:
        rows = [(int(i), {"label": CODES[c]}) for i, c in re.findall(r"(\d+)\s*([dpsgo])", data["codes"])]
    return [{"order_id": order_ids[i], **r} for i, r in dict(rows).items() if 0 <= i < len(order_ids)]  # dict: a repeated number counts once


def chunks(df: pd.DataFrame):
    for start in range(0, len(df), CHUNK):
        part = df.iloc[start:start + CHUNK]
        yield start, part.order_id.tolist(), part.comment.tolist()


def label_ollama(df: pd.DataFrame, model: str = "qwen3:8b", translate: bool = False) -> pd.DataFrame:
    """Free local run, for testing the pipeline on a small sample (slow on CPU)."""
    rows = []
    for _, ids, comments in chunks(df):
        body = {"model": model, "stream": False, "think": False, "format": schema(translate),
                "options": {"temperature": 0, "num_ctx": 16384},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt(comments, translate)}]}
        req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            rows += parse(json.load(r)["message"]["content"], ids, translate)
    return pd.DataFrame(rows)


def submit_batch(df: pd.DataFrame, model: str, translate: bool = False) -> str:
    """Send all reviews to the Anthropic Batch API; returns the batch id."""
    import anthropic

    requests = [{
        "custom_id": f"chunk-{start}",
        "params": {"model": model, "max_tokens": 8000 if translate else 500,
                   "thinking": {"type": "disabled"},
                   "system": SYSTEM,
                   "output_config": {"format": {"type": "json_schema", "schema": schema(translate)}},
                   "messages": [{"role": "user", "content": prompt(comments, translate)}]},
    } for start, _, comments in chunks(df)]
    return anthropic.Anthropic().messages.batches.create(requests=requests).id


def collect_batch(batch_id: str, df: pd.DataFrame, translate: bool = False) -> pd.DataFrame:
    """Download results of an ended batch; reviews from failed chunks are reported and left out."""
    import anthropic

    client = anthropic.Anthropic()
    ids = {f"chunk-{start}": ids for start, ids, _ in chunks(df)}
    rows, failed = [], []
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            failed.append(res.custom_id)
            continue
        text = next(b.text for b in res.result.message.content if b.type == "text")
        parsed = parse(text, ids[res.custom_id], translate)
        if not parsed:
            failed.append(res.custom_id)
        rows += parsed
    if failed:
        print(f"{len(failed)} chunks failed: {failed[:5]}")
    return pd.DataFrame(rows)
