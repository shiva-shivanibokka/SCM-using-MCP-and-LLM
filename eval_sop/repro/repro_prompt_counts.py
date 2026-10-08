"""Check the data counts stated in the agent system prompt against the data,
for the base commit and for this branch. Run from repo root."""
import re, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
truth = {
    "stores": len(pd.read_csv(D / "huft_stores.csv")),
    "SKUs (catalog)": len(pd.read_csv(D / "huft_products.csv")),
    "customers": len(pd.read_csv(D / "huft_customers.csv")),
    "demand rows": len(pd.read_csv(D / "huft_daily_demand.csv")),
    "transactions": len(pd.read_csv(D / "huft_sales_transactions.csv")),
    "returns": len(pd.read_csv(D / "huft_returns.csv")),
    "supplier reviews": len(pd.read_csv(D / "huft_supplier_performance.csv")),
}
sys.path.insert(0, str(ROOT))
from mcp_server.server import MCP_TOOLS  # noqa: E402
truth["tools"] = len(MCP_TOOLS)
pats = {
    "stores": r"with (\d[\d,]*) stores across India",
    "SKUs (catalog)": r"huft_products\.csv\s+— (\d[\d,]*) SKUs",
    "customers": r"SKUs, (\d[\d,]*) customers",
    "demand rows": r"(\d[\d,]*) rows \| \d+ SKUs",
    "transactions": r"huft_sales_transactions\.csv — (\d[\d,]*) transactions",
    "returns": r"huft_returns\.csv\s+— (\d[\d,]*) returns",
    "supplier reviews": r"— (\d[\d,]*) monthly supplier reviews",
    "tools": r"You have access to (\d+) powerful tools",
}
for tag, src in [("BASE 65c1c06", subprocess.check_output(["git", "-C", str(ROOT), "show", "65c1c06:agent/agent.py"]).decode("utf-8")),
                 ("THIS BRANCH", (ROOT / "agent/agent.py").read_text(encoding="utf-8"))]:
    bad = 0
    print(f"== {tag}")
    for k, pat in pats.items():
        m = re.search(pat, src)
        v = int(m.group(1).replace(",", "")) if m else None
        ok = v == truth[k]; bad += not ok
        print(f"  {k:18s} prompt={v!s:>8}  data={truth[k]:>8}  {'OK' if ok else 'MISMATCH'}")
    print(f"  -> {bad} mismatches")
