"""Recursive Language Model (Zhang, Kraska & Khattab), built from scratch.

The context never goes into the prompt. It lives in a Python REPL as the variable `context`;
the root model only sees the question and a short description, and works by writing code:
  depth 0  code only (string matching, counting)
  depth 1  code + llm_query(prompt): a plain sub-model call, for judgments code can't make
  depth 2  same, but each llm_query is itself an RLM (depth 1) with the prompt as its context
The root ends with FINAL(answer) or FINAL_VAR(variable_name).
"""
import builtins
import io
import re
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from .labels import LABEL_RULES
from .llm import LLM, Reply

CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml").read_text())
ALLOWED_IMPORTS = {"re", "json", "math", "collections", "statistics", "itertools", "functools"}
SAFE_BUILTINS = {name: getattr(builtins, name) for name in [
    "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "ord", "pow", "print", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError", "ZeroDivisionError", "StopIteration"]}
CODE = re.compile(r"```repl\s*\n(.*?)```", re.S)
FINAL_LINE = re.compile(r"^FINAL\((.*)\)[ \t]*$", re.M)
FINAL_VAR_LINE = re.compile(r"^FINAL_VAR\((\w+)\)[ \t]*$", re.M)

SYSTEM = """You answer a question about a large text you cannot see directly. It is stored in a Python REPL
as the variable `context` ({chars:,} characters, {lines:,} lines). Each line is one order record:
[order ID] seller=ID (state/city) customer=state/city category=... purchased/estimated/delivered dates (delivery status) review=score/5 "comment in Portuguese"
Questions about complaints use these review labels:
{rules}

Write Python inside ```repl ... ``` blocks. After each reply you see what the code printed (cut to {out_chars} characters),
so print what you need. Variables persist between blocks. Allowed imports: {imports}. No files or network.
{tools}
After your code blocks, stop and wait: you will be shown their real output. Never write the output yourself.
Work step by step: look at the data first (e.g. print(context[:1500])), then compute the answer with code.
When you are sure, write on its own line FINAL(answer), or FINAL_VAR(name) to return a variable.
Give the answer in exactly the format the question asks for."""

TOOLS = {
    0: "There is no language model inside the REPL: use code only (string matching, regex, counting).",
    1: """You can call llm_query(prompt) -> str. It sends `prompt` to a language model that reads at most {sub_chars:,}
characters at once. Use it for judgments code cannot make, such as reading review comments. Put many records in one
call (e.g. 40 lines with the instruction to answer per line) and loop over chunks, instead of one call per record.""",
}


class Timeout(Exception):
    pass


class RLM:
    def __init__(self, root: LLM, sub: LLM | None = None, depth: int = 1, run_budget: float = CFG["budget_usd_per_run"]):
        self.root, self.sub, self.depth, self.run_budget = root, sub or root, depth, run_budget
        self.cfg = CFG["rlm"]
        self.sub_chars = int(self.sub.num_ctx * 1.8)  # ~2 characters per token for this data, minus room for the reply

    # ---------- the REPL ----------
    def _namespace(self, context: str) -> dict:
        def safe_import(name, *args, **kwargs):
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                raise ImportError(f"import of '{name}' is not allowed")
            return __import__(name, *args, **kwargs)

        ns = {"__builtins__": {**SAFE_BUILTINS, "__import__": safe_import}, "context": context}
        ns["FINAL"] = lambda answer: ns.__setitem__("__final__", str(answer))
        ns["FINAL_VAR"] = lambda name: ns.__setitem__("__final__", str(ns[name]))
        if self.depth >= 1:
            ns["llm_query"] = self._llm_query
        return ns

    def _execute(self, code: str, ns: dict) -> str:
        """Run one code block; return what it printed, or the error.
        Time limit: a trace hook checks the clock on every bytecode instruction of the model's own code
        (compiled as "<rlm>"); library code it calls, such as llm_query, is not traced and runs at full speed."""
        deadline = time.monotonic() + self.cfg["code_timeout_s"]

        def check(frame, event, arg):
            if time.monotonic() > deadline:
                raise Timeout(f"code took longer than {self.cfg['code_timeout_s']}s")
            return check

        def on_call(frame, event, arg):
            if frame.f_code.co_filename != "<rlm>":
                return None
            frame.f_trace_opcodes = True  # per instruction: even a one-line `while True: pass` is caught
            return check

        out = io.StringIO()
        sys.settrace(on_call)
        try:
            with redirect_stdout(out):
                exec(compile(code, "<rlm>", "exec"), ns)
        except Exception as e:  # the model sees its own errors and can fix them
            out.write(f"\n{type(e).__name__}: {e}" if isinstance(e, Timeout) else "\n" + traceback.format_exc(limit=1))
        finally:
            sys.settrace(None)
        text = out.getvalue()
        limit = self.cfg["output_chars"]
        return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit:,} more characters cut]"

    # ---------- sub-calls ----------
    def _llm_query(self, prompt: str) -> str:
        if len(self.sub_calls) >= self.cfg["max_sub_calls"]:
            return "ERROR: sub-call limit reached; answer with what you have."
        if self.cost >= self.run_budget:
            return "ERROR: budget for this question used up; answer with what you have."
        if len(prompt) > self.sub_chars:
            return f"ERROR: prompt has {len(prompt):,} characters, the limit is {self.sub_chars:,}. Split it into smaller chunks."
        if self.depth >= 2:  # recursion: the sub-call is itself an RLM over the prompt
            child = RLM(self.sub, self.sub, self.depth - 1, self.run_budget - self.cost)
            reply = child.answer("Carry out the instructions at the start of `context` and FINAL the result.", prompt)
        else:
            reply = self.sub.chat(prompt)
        self._add(reply)
        self.sub_calls.append({"prompt_chars": len(prompt), "reply": reply.text[:500], "cost": reply.cost})
        return reply.text.rsplit("ANSWER:", 1)[-1].strip() if self.depth >= 2 else reply.text

    def _add(self, reply: Reply):
        self.tokens_in += reply.input_tokens
        self.tokens_out += reply.output_tokens
        self.cost += reply.cost

    # ---------- the root loop ----------
    def answer(self, question: str, context: str) -> Reply:
        start = time.time()
        self.tokens_in = self.tokens_out = 0
        self.cost, self.sub_calls, trace = 0.0, [], []
        ns = self._namespace(context)
        system = SYSTEM.format(chars=len(context), lines=context.count("\n") + 1, rules=LABEL_RULES,
                               out_chars=self.cfg["output_chars"], imports=", ".join(sorted(ALLOWED_IMPORTS)),
                               tools=TOOLS[min(self.depth, 1)].format(sub_chars=self.sub_chars))
        transcript = f"Question: {question}"
        final = None
        for step in range(self.cfg["max_iterations"]):
            last = step == self.cfg["max_iterations"] - 1 or self.cost >= self.run_budget
            prompt = transcript + ("\n\nNo more turns left: write FINAL(answer) now." if last else "")
            reply = self.root.chat(prompt, system)
            self._add(reply)
            n_sub = len(self.sub_calls)
            blocks = CODE.findall(reply.text)
            outputs = [self._execute(code, ns) for code in blocks]
            trace.append({"step": step, "response": reply.text, "outputs": outputs,
                          "sub_calls": self.sub_calls[n_sub:]})
            # A FINAL typed next to code blocks is ignored: the model has not seen their real output yet
            # (models sometimes invent the output and answer from it). FINAL() run inside the code is fine.
            final = ns.get("__final__") or (None if blocks else self._final_in_text(reply.text, ns))
            if final is not None or last:
                break
            transcript += f"\n\n--- Your reply {step + 1} ---\n{reply.text}\n--- REPL output ---\n" + (
                "\n".join(outputs) or "(no ```repl``` block found: write code, or FINAL(answer) if done)")
        return Reply(f"ANSWER: {final if final is not None else ''}", self.tokens_in, self.tokens_out,
                     self.cost, time.time() - start, trace)

    @staticmethod
    def _final_in_text(text: str, ns: dict) -> str | None:
        """FINAL written as plain text outside any code block."""
        outside = CODE.sub("", text)
        if m := FINAL_VAR_LINE.search(outside):
            return str(ns.get(m.group(1), f"(variable {m.group(1)} not found)"))
        if m := FINAL_LINE.search(outside):
            return m.group(1).strip()
        return None
