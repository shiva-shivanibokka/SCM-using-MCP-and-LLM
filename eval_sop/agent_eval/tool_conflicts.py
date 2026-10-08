"""LLM-free measurement: do overlapping MCP tools agree with each other and
with pandas ground truth? Calls the tools in-process (same dispatch the agent
uses), parses the number each tool reports, before and after the stockout fix.

  python -m eval_sop.agent_eval.tool_conflicts
Writes results/tool_conflicts.json and prints a markdown table.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["DATABASE_URL"] = ""

import intelligence.stockout as ST  # noqa: E402
import mcp_server.server as S  # noqa: E402

QS = {json.loads(l)["id"]: json.loads(l) for l in (Path(__file__).parent / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
ORIG = ST.predict_stockouts


def set_variant(v):
    if v == "before":
        ST.predict_stockouts = lambda inv, safety_stock_days=7, risk_filter=None, **kw: ORIG(
            inv, safety_stock_days=safety_stock_days, risk_filter=risk_filter, velocity_window_days=1)
    else:
        ST.predict_stockouts = ORIG
    S.invalidate_tool_cache()


async def call(name, args):
    return await S.call_tool_direct(name, args)


def f(pat, text, grp=1):
    m = re.search(pat, text, re.S)
    return float(m.group(grp).replace(",", "")) if m else None


async def sku_cover(sku):
    out = {}
    pred = await call("get_stockout_prediction", {"top_n": 200})
    m = re.search(rf"\| {sku} \|[^|]*\|\s*([\d,]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.∞]+)\s*\|", pred)
    out["get_stockout_prediction"] = float(m.group(3)) if m and m.group(3) != "∞" else None
    out["get_sku_360"] = f(r"Days of Supply:\s*([\d.]+)", await call("get_sku_360", {"sku_id": sku}))
    out["get_inventory_status"] = f(r"Days of Supply:\s*([\d.]+)", await call("get_inventory_status", {"sku_id": sku}))
    risk = await call("get_stockout_risk", {"days": 30})
    m = re.search(rf"\] {sku} –.*?Stocks out in : ([\d.]+) days", risk, re.S)
    out["get_stockout_risk"] = float(m.group(1)) if m else None  # None = SKU not listed as at risk
    return out


async def critical_count():
    out = {}
    pred = await call("get_stockout_prediction", {"top_n": 5})
    out["get_stockout_prediction"] = f(r"🔴 (\d+) critical", pred)
    dash = await call("get_supply_chain_dashboard", {})
    out["get_supply_chain_dashboard"] = f(r"CRITICAL\s*:\s*(\d+)", dash)
    rl = await call("get_reorder_list", {})
    out["get_reorder_list (SKUs needing reorder)"] = 0.0 if "No reorders needed" in rl else float(len(re.findall(r"^\s*\[", rl, re.M)) or len(re.findall(r"SKU", rl)))
    return out


async def lowest_cover():
    out = {}
    pred = await call("get_stockout_prediction", {"top_n": 1})
    m = re.search(r"\| (\w+_\d+) \|", pred.split("|------|")[-1]) or re.search(r"\n\| ([A-Z]+_[A-Z0-9_]+) \|", pred)
    out["get_stockout_prediction"] = m.group(1) if m else None
    risk = await call("get_stockout_risk", {"days": 14})
    m = re.search(r"\] ([A-Z]+_[A-Z0-9_]+) –", risk)
    out["get_stockout_risk"] = m.group(1) if m else None
    return out


async def supplier_otd(name):
    t = await call("get_supplier_lead_time_tracker", {})
    m = re.search(rf"{re.escape(name)}\s*\n\s*OTD: ([\d.]+)%", t)
    return {"get_supplier_lead_time_tracker": float(m.group(1)) if m else None}


async def inventory(sku):
    pred = await call("get_stockout_prediction", {"top_n": 200})
    m = re.search(rf"\| {sku} \|[^|]*\|\s*([\d,]+)\s*\|", pred)
    return {"get_stockout_prediction (store-level sum)": float(m.group(1).replace(",", "")) if m else None,
            "get_sku_360": f(r"Stock\s*:\s*([\d,]+) units", await call("get_sku_360", {"sku_id": sku})),
            "get_inventory_status": f(r"Current Inventory:\s*([\d,]+)", await call("get_inventory_status", {"sku_id": sku}))}


def truth(qid, i=0):
    return QS[qid]["parts"][i]["value"]


async def main():
    rows = []
    for v in ["before", "after"]:
        set_variant(v)
        for qid, sku in [("C06", "EXT_059"), ("C07", "EXT_077"), ("C08", "FOOD_D011")]:
            for tool, val in (await sku_cover(sku)).items():
                rows.append({"variant": v, "qid": qid, "quantity": f"days of cover {sku}", "tool": tool,
                             "tool_value": val, "truth": round(truth(qid), 2), "tol": QS[qid]["parts"][0]["tol"]})
        for tool, val in (await critical_count()).items():
            rows.append({"variant": v, "qid": "C09", "quantity": "# critical SKUs", "tool": tool,
                         "tool_value": val, "truth": truth("C09"), "tol": 0.01})
        for tool, val in (await lowest_cover()).items():
            rows.append({"variant": v, "qid": "C10", "quantity": "SKU with lowest cover", "tool": tool,
                         "tool_value": val, "truth": truth("C10"), "tol": None})
        for qid, sup in [("C12", "Trixie India"), ("A05", "Royal Canin India")]:
            for tool, val in (await supplier_otd(sup)).items():
                rows.append({"variant": v, "qid": qid, "quantity": f"2025 OTD % {sup}", "tool": tool,
                             "tool_value": val, "truth": round(truth(qid), 2), "tol": QS[qid]["parts"][0]["tol"]})
        for tool, val in (await inventory("EXT_059")).items():
            rows.append({"variant": v, "qid": "C14", "quantity": "inventory EXT_059", "tool": tool,
                         "tool_value": val, "truth": truth("C14"), "tol": 0.5})
    for r in rows:
        if r["tol"] is None:
            r["agrees_with_truth"] = r["tool_value"] == r["truth"]
        else:
            r["agrees_with_truth"] = r["tool_value"] is not None and abs(r["tool_value"] - r["truth"]) <= r["tol"]
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "tool_conflicts.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("| variant | Q | quantity | tool | tool value | ground truth | agrees |\n|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['qid']} | {r['quantity']} | {r['tool']} | {r['tool_value']} | {r['truth']} | {'yes' if r['agrees_with_truth'] else 'NO'} |")
    for v in ["before", "after"]:
        rs = [r for r in rows if r["variant"] == v]
        print(f"{v}: {sum(r['agrees_with_truth'] for r in rs)}/{len(rs)} tool readings agree with ground truth")


if __name__ == "__main__":
    asyncio.run(main())
