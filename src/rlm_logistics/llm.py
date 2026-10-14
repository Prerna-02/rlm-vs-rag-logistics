"""One client for every model (local Ollama, OpenRouter, Anthropic). Every call returns the text plus
its token usage, cost and time, so all methods are measured the same way."""
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
LONG_PROMPT = 100_000  # Anthropic charges a higher rate above this many input tokens


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    seconds: float


def post_json(url: str, body: dict, headers: dict | None = None, timeout: int = 3600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class LLM:
    def __init__(self, provider: str, model: str, price=(0, 0), long_price=None, num_ctx: int = 8192,
                 base_url: str = "http://localhost:11434"):
        self.provider, self.model, self.num_ctx, self.base_url = provider, model, num_ctx, base_url
        self.price, self.long_price = price, long_price or price  # $ per 1M (input, output) tokens

    @classmethod
    def from_config(cls, role: str) -> "LLM":
        return cls(**yaml.safe_load(CONFIG.read_text())[role])

    def chat(self, prompt: str, system: str = "", schema: dict | None = None, max_tokens: int = 4000) -> Reply:
        start = time.time()
        text, tin, tout = getattr(self, f"_{self.provider}")(prompt, system, schema, max_tokens)
        p_in, p_out = self.long_price if tin > LONG_PROMPT else self.price
        return Reply(text, tin, tout, (tin * p_in + tout * p_out) / 1e6, time.time() - start)

    def _ollama(self, prompt, system, schema, max_tokens):
        r = post_json(f"{self.base_url}/api/chat", {
            "model": self.model, "stream": False, "think": False, "format": schema,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
        return r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0)

    def _openrouter(self, prompt, system, schema, max_tokens):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        r = post_json("https://openrouter.ai/api/v1/chat/completions", body,
                      {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]

    def _anthropic(self, prompt, system, schema, max_tokens):
        import anthropic

        kwargs = {"output_config": {"format": {"type": "json_schema", "schema": schema}}} if schema else {}
        with anthropic.Anthropic().messages.stream(model=self.model, max_tokens=max_tokens, system=system,
                                                   messages=[{"role": "user", "content": prompt}], **kwargs) as s:
            msg = s.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens
