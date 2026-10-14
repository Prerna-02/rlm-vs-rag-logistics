"""Hard spending cap. Every paid call is appended to a local ledger file, so the total survives restarts;
once the total reaches the limit, any further paid call raises BudgetExceeded before it is sent.

The limit is personal, so it lives in .env (BUDGET_USD), not in the repo. Without it, paid calls are refused."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "results" / "spend.jsonl"


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, limit: float | None = None, ledger: Path = LEDGER):
        self.limit = float(os.environ.get("BUDGET_USD", 0)) if limit is None else limit
        self.ledger = ledger
        self.spent = sum(json.loads(l)["cost"] for l in ledger.read_text().splitlines()) if ledger.exists() else 0.0

    def check(self):
        """Call before a paid request: refuses once the total has reached the limit."""
        if self.spent >= self.limit:
            raise BudgetExceeded("spending limit reached (set BUDGET_USD in .env)")

    def charge(self, cost: float, note: str = ""):
        if cost <= 0:
            return
        self.spent += cost
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cost": cost, "note": note}) + "\n")
