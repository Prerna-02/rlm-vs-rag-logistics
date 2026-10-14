"""RLM loop tests with a scripted fake model: no Ollama, no API, runs in a second."""
import pytest

from rlm_logistics.budget import Budget, BudgetExceeded
from rlm_logistics.llm import LLM, Reply
from rlm_logistics.rlm import RLM


class Scripted:
    """Stands in for an LLM: returns the given replies in order and records the prompts it saw."""

    def __init__(self, *replies, num_ctx=8192, cost=0.0):
        self.replies, self.prompts, self.num_ctx, self.model, self.cost = list(replies), [], num_ctx, "fake", cost

    def chat(self, prompt, system="", schema=None, max_tokens=4000):
        self.prompts.append(prompt)
        return Reply(self.replies.pop(0), 10, 5, self.cost, 0.0)


CTX = "line one late\nline two\nline three late"


def test_variables_persist_and_final_var_returns_them():
    root = Scripted("```repl\nn = sum('late' in l for l in context.split('\\n'))\nprint(n)\n```",
                    "FINAL_VAR(n)")
    reply = RLM(root, depth=0).answer("How many late?", CTX)
    assert reply.text == "ANSWER: 2"
    assert "2" in root.prompts[1]  # the second turn saw the printed output
    assert len(reply.trace) == 2


def test_depth1_llm_query_reaches_the_sub_model():
    root = Scripted("```repl\nr = llm_query('classify: ' + context.split('\\n')[0])\nFINAL(r)\n```")
    sub = Scripted("delivery")
    reply = RLM(root, sub, depth=1).answer("q", CTX)
    assert reply.text == "ANSWER: delivery" and sub.prompts == ["classify: line one late"]
    assert reply.input_tokens == 20  # root + sub usage are both counted


def test_depth0_has_no_sub_model():
    root = Scripted("```repl\nllm_query('x')\n```", "FINAL(none)")
    RLM(root, depth=0).answer("q", CTX)
    assert "NameError" in root.prompts[1]


def test_depth2_sub_call_is_itself_an_rlm():
    root = Scripted("```repl\nFINAL(llm_query('count lines'))\n```")
    sub = Scripted("```repl\nFINAL(len(context.split()))\n```")  # the child RLM sees the prompt as its context
    assert RLM(root, sub, depth=2).answer("q", CTX).text == "ANSWER: 2"


def test_sandbox_blocks_imports_files_and_runaway_loops():
    root = Scripted("```repl\nimport os\n```\n```repl\nopen('x.txt', 'w')\n```", "FINAL(done)")
    RLM(root, depth=0).answer("q", CTX)
    assert "not allowed" in root.prompts[1] and "NameError" in root.prompts[1]
    loop = RLM(Scripted("```repl\nwhile True: pass\n```", "FINAL(x)"), depth=0)
    loop.cfg = {**loop.cfg, "code_timeout_s": 0.2}
    assert "Timeout" in loop.answer("q", CTX).trace[0]["outputs"][0]


def test_long_sub_prompt_is_refused_with_advice():
    root = Scripted("```repl\nprint(llm_query('x' * 100_000))\n```", "FINAL(ok)")
    sub = Scripted(num_ctx=1000)
    RLM(root, sub, depth=1).answer("q", CTX)
    assert "Split it" in root.prompts[1] and sub.prompts == []


def test_iteration_cap_forces_a_final_turn():
    rlm = RLM(Scripted(*["thinking..."] * 3), depth=0)
    rlm.cfg = {**rlm.cfg, "max_iterations": 3}
    reply = rlm.answer("q", CTX)
    assert len(reply.trace) == 3 and reply.text == "ANSWER: "


def test_budget_blocks_paid_calls_once_spent(tmp_path):
    budget = Budget(limit=0.01, ledger=tmp_path / "spend.jsonl")
    budget.charge(0.02, "earlier run")
    llm = LLM("anthropic", "claude-haiku-5-5", budget=budget)
    with pytest.raises(BudgetExceeded):
        llm.chat("hi")
    assert Budget(limit=0.01, ledger=tmp_path / "spend.jsonl").spent == 0.02  # the total survives a restart


def test_final_written_next_to_code_waits_for_the_real_output():
    root = Scripted("```repl\nn = 3 + 4\n```\noutput: 99\nFINAL(99)", "FINAL_VAR(n)")
    assert RLM(root, depth=0).answer("q", CTX).text == "ANSWER: 7"


def test_budget_without_a_limit_refuses_paid_calls(tmp_path, monkeypatch):
    monkeypatch.delenv("BUDGET_USD", raising=False)
    with pytest.raises(BudgetExceeded):
        Budget(ledger=tmp_path / "spend.jsonl").check()
