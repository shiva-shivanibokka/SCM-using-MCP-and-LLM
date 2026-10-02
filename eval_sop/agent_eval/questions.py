"""Build the agent-eval question bank with ground truth computed directly from
the raw CSVs (pandas only -- no project tool code is used for ground truth).

Run:  python -m eval_sop.agent_eval.questions   (from the repo root)
Writes eval_sop/agent_eval/questions.jsonl

Every question states its definition explicitly (window, metric, source) so
the ground truth is unambiguous. Each record:
  id, category (lookup | aggregation | multihop | conflict), question,
  parts: [ {kind: number|entity, value, tol (abs, numbers), aliases (entities)} ],
  distractors: values a known-wrong source would give (stale prompt, buggy tool),
  acceptable_tools: tools that can legitimately produce the answer,
  fix_relevant: True if the stockout / stale-prompt fix should matter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "questions.jsonl"

GENERIC = ["run_sql_query", "python_repl"]  # always-legitimate ad-hoc tools


def load():
    d = {}
    d["p"] = pd.read_csv(DATA / "huft_products.csv")
    d["s"] = pd.read_csv(DATA / "huft_stores.csv")
    d["c"] = pd.read_csv(DATA / "huft_customers.csv")
    d["pr"] = pd.read_csv(DATA / "huft_promotions.csv")
    d["t"] = pd.read_csv(DATA / "huft_sales_transactions.csv", parse_dates=["date"])
    d["r"] = pd.read_csv(DATA / "huft_returns.csv", parse_dates=["return_date"])
    d["sp"] = pd.read_csv(DATA / "huft_supplier_performance.csv")
    d["cc"] = pd.read_csv(DATA / "huft_cold_chain.csv", parse_dates=["date"])
    d["dd"] = pd.read_csv(DATA / "huft_daily_demand.csv", parse_dates=["date"])
    d["sdi"] = pd.read_csv(DATA / "store_daily_inventory.csv", parse_dates=["date"])
    return d


def num(v, tol):
    return {"kind": "number", "value": float(v), "tol": float(tol)}


def ent(v, aliases=None):
    al = [str(v)] + list(aliases or [])
    return {"kind": "entity", "value": str(v), "aliases": al}


def cover_table(dd: pd.DataFrame) -> pd.DataFrame:
    """SKU-level days of cover = latest inventory / mean demand over the last
    30 days (inclusive of the latest date), from huft_daily_demand.csv."""
    last = dd["date"].max()
    w = dd[dd["date"] > last - pd.Timedelta(days=30)]
    avg30 = w.groupby("sku_id")["demand"].mean()
    inv = dd[dd["date"] == last].set_index("sku_id")[["inventory", "lead_time_days", "name"]]
    out = inv.join(avg30.rename("avg30"))
    out["cover"] = out["inventory"] / out["avg30"]
    return out


def store_level_lastday(sdi: pd.DataFrame) -> pd.DataFrame:
    """What the ORIGINAL get_stockout_prediction computes: latest row per
    (store, SKU), velocity = that single day's demand summed across stores."""
    snap = sdi.sort_values("date").groupby(["store_id", "sku_id"]).tail(1)
    g = snap.groupby("sku_id").agg(inv=("inventory", "sum"), vel=("demand", "sum"),
                                   lead=("lead_time_days", "max"))
    g["dtz"] = g["inv"] / g["vel"]
    return g


def build() -> list[dict]:
    d = load()
    p, s, c, pr, t, r, sp, cc, dd, sdi = (d[k] for k in
                                          ["p", "s", "c", "pr", "t", "r", "sp", "cc", "dd", "sdi"])
    t25 = t[t["date"].dt.year == 2025]
    t24 = t[t["date"].dt.year == 2024]
    sp25 = sp[sp["review_month"].str.startswith("2025")]
    cov = cover_table(dd)
    old = store_level_lastday(sdi)
    Q: list[dict] = []

    def add(qid, cat, q, parts, distractors=None, tools=None, fix=False, note=""):
        Q.append({"id": qid, "category": cat,
                  "question": q.strip(),
                  "parts": parts, "distractors": distractors or [],
                  "acceptable_tools": sorted(set((tools or []) + GENERIC)),
                  "fix_relevant": fix, "note": note})

    # ------------------------------------------------------------------ lookups
    sku = "EXT_059"; row = p.set_index("sku_id").loc[sku]
    add("L01", "lookup", f"What is the retail price (price_inr) of SKU {sku}?",
        [num(row.price_inr, 0.5)], tools=["get_sku_360", "get_inventory_status"])
    row = p.set_index("sku_id").loc["FOOD_D011"]
    add("L02", "lookup", "Which supplier supplies SKU FOOD_D011?",
        [ent(row.supplier, ["Sara's Kitchen"])], tools=["get_sku_360", "get_inventory_status"])
    row = p.set_index("sku_id").loc["FOOD_D001"]
    add("L03", "lookup", "What is the catalog lead time in days (lead_time_days in the product catalog) for SKU FOOD_D001?",
        [num(row.lead_time_days, 0.01)], tools=["get_sku_360", "get_inventory_status"])
    st = s.set_index("store_id").loc["ST001"]
    add("L04", "lookup", "In which city is store ST001 located?", [ent(st.city)],
        tools=["get_store_inventory_breakdown", "get_franchise_inventory_comparison"])
    row = p.set_index("sku_id").loc["FOOD_D003"]
    add("L05", "lookup", "What is the margin_pct of SKU FOOD_D003 in the product catalog?",
        [num(row.margin_pct, 0.05)], tools=["get_sku_360"])
    promo = pr.iloc[0]
    add("L06", "lookup", f"What discount percentage did the promotion '{promo['name']}' offer?",
        [num(promo.discount_pct, 0.05)], tools=["get_promotion_inventory_impact"])
    sku = p.sort_values("price_inr", ascending=False).iloc[0]
    add("L07", "lookup", "Which SKU has the highest retail price (price_inr) in the product catalog? Give the sku_id.",
        [ent(sku.sku_id)])
    row = p.set_index("sku_id").loc["GROM_004"]
    add("L08", "lookup", "What brand is SKU GROM_004?", [ent(row.brand, ["HUFT"])],
        tools=["get_sku_360", "get_inventory_status"])
    add("L09", "lookup", "How many stores are located in the city of Bengaluru?",
        [num((s.city == "Bengaluru").sum(), 0.01)],
        tools=["get_store_inventory_breakdown", "get_franchise_inventory_comparison"])
    add("L10", "lookup", "How many SKUs in the product catalog are cold-chain items (is_cold_chain = true)?",
        [num(p.is_cold_chain.sum(), 0.01)], tools=["get_cold_chain_monitor"])
    row = p.set_index("sku_id").loc["EXT_066"]
    add("L11", "lookup", "Which product category does SKU EXT_066 belong to?", [ent(row.category)],
        tools=["get_sku_360", "get_inventory_status"])
    add("L12", "lookup", "How many promotions are recorded in the promotions data?",
        [num(len(pr), 0.01)], tools=["get_promotion_inventory_impact"])

    # ------------------------------------------------------------- aggregations
    add("A01", "aggregation", "What was the total net revenue (sum of net_revenue_inr) across all sales transactions in calendar year 2025, in INR?",
        [num(t25.net_revenue_inr.sum(), t25.net_revenue_inr.sum() * 0.01)],
        tools=["get_channel_revenue_attribution", "get_inventory_financial_summary"])
    ch = t25.groupby("channel").net_revenue_inr.sum()
    add("A02", "aggregation", "Which sales channel had the highest net revenue in 2025, and what percentage of total 2025 net revenue did it account for?",
        [ent(ch.idxmax()), num(100 * ch.max() / ch.sum(), 0.5)],
        tools=["get_channel_revenue_attribution"])
    cat = t24.groupby("category").quantity.sum()
    add("A03", "aggregation", "Which product category sold the most units (sum of quantity in sales transactions) in 2024?",
        [ent(cat.idxmax())], tools=["compare_categories"])
    rr = r.return_reason.value_counts()
    add("A04", "aggregation", "What is the most common return reason across all returns, and how many returns had that reason?",
        [ent(rr.idxmax()), num(rr.max(), 0.01)], tools=["get_return_rate_analysis"])
    otd = sp25.groupby("supplier_name").on_time_delivery_pct.mean()
    add("A05", "aggregation", "What was Royal Canin India's average on-time delivery percentage across its 2025 monthly reviews?",
        [num(otd["Royal Canin India"], 0.1)], distractors=[round(sp[sp.supplier_name == "Royal Canin India"].on_time_delivery_pct.mean(), 1)],
        tools=["get_supplier_lead_time_tracker", "get_supplier_fill_rate_trend", "get_supplier_ranking"])
    add("A06", "aggregation", "Which supplier had the lowest average on-time delivery percentage across its 2025 monthly reviews?",
        [ent(otd.idxmin())], tools=["get_supplier_lead_time_tracker", "get_supplier_fill_rate_trend"])
    reg = s.region.value_counts()
    add("A07", "aggregation", "Which region has the most stores, and how many stores does it have?",
        [ent(reg.idxmax()), num(reg.max(), 0.01)],
        tools=["get_franchise_inventory_comparison", "get_store_inventory_breakdown"])
    add("A08", "aggregation", "How many temperature-breach records (temp_breach = true) are there in the full cold-chain monitoring history?",
        [num(cc.temp_breach.sum(), 0.01)], tools=["get_cold_chain_monitor"])
    last = dd.date.max()
    w30 = dd[dd.date > last - pd.Timedelta(days=30)]
    v = w30[w30.sku_id == "FOOD_D001"].demand.mean()
    add("A09", "aggregation", f"What was the average daily demand of SKU FOOD_D001 over the last 30 days of the demand data (the 30 days ending {last.date()})?",
        [num(v, max(0.5, v * 0.02))], tools=["get_sku_360", "get_inventory_status", "get_demand_forecast"])
    q = t24[t24.sku_id == "GROM_004"].quantity.sum()
    add("A10", "aggregation", "How many units of SKU GROM_004 were sold in 2024 (sum of quantity in sales transactions)?",
        [num(q, max(1, q * 0.005))])
    seg = c.segment.value_counts()
    add("A11", "aggregation", "How many customers are in the 'Loyal Premium' segment?",
        [num(seg["Loyal Premium"], 0.01)], tools=["get_customer_segmentation_insights"])
    city = t25.groupby("city").net_revenue_inr.sum()
    add("A12", "aggregation", "Which city generated the highest net revenue in 2025?", [ent(city.idxmax())],
        tools=["get_channel_revenue_attribution", "get_store_level_demand_intelligence"])
    add("A13", "aggregation", "How many SKUs in the product catalog are in the 'Dog Food' category?",
        [num((p.category == "Dog Food").sum(), 0.01)], tools=["compare_categories"])
    ref = r[r.return_date.dt.year == 2025].refund_inr.sum()
    add("A14", "aggregation", "What was the total refund amount (sum of refund_inr) for returns dated in 2025, in INR?",
        [num(ref, ref * 0.01)], tools=["get_return_rate_analysis"])
    mon = t25.groupby(t25.date.dt.month).net_revenue_inr.sum()
    mname = pd.Timestamp(2025, int(mon.idxmax()), 1).strftime("%B")
    add("A15", "aggregation", "Which calendar month of 2025 had the highest total net revenue?",
        [ent(mname, [mname[:3]])])
    ltv = c.groupby("segment").lifetime_value_inr.mean()
    add("A16", "aggregation", "Which customer segment has the highest average lifetime_value_inr?",
        [ent(ltv.idxmax())], tools=["get_customer_segmentation_insights"])

    # ----------------------------------------------------------------- multihop
    skurev = t25.groupby("sku_id").net_revenue_inr.sum()
    top = skurev.idxmax(); sup = p.set_index("sku_id").loc[top, "supplier"]
    dr = sp25[sp25.supplier_name == sup].defect_rate_pct.mean()
    add("M01", "multihop", "Identify the SKU with the highest 2025 net revenue. Who is its supplier, and what was that supplier's average defect_rate_pct across its 2025 monthly reviews?",
        [ent(sup), num(dr, 0.05)], tools=["get_supplier_lead_time_tracker", "get_supplier_fill_rate_trend", "get_brand_performance"],
        note=f"top SKU {top}")
    rc = r.sku_id.value_counts()
    add("M02", "multihop", "Which SKU has the largest number of return records, and which product category is it in?",
        [ent(rc.idxmax()), ent(p.set_index("sku_id").loc[rc.idxmax(), "category"])],
        tools=["get_return_rate_analysis"])
    t25s = t25.merge(s[["store_id", "region"]], on="store_id", how="left")
    rg = t25s.groupby("region").net_revenue_inr.sum()
    add("M03", "multihop", "Joining sales transactions to the stores table on store_id, which store region generated the highest net revenue in 2025?",
        [ent(rg.idxmax())], tools=["get_franchise_inventory_comparison"])
    sold = t.groupby("sku_id").quantity.sum(); ret = r.groupby("sku_id").quantity_returned.sum()
    rate = (ret / sold).dropna()
    hs = rate.idxmax()
    reason = r[r.sku_id == hs].return_reason.value_counts().idxmax()
    add("M04", "multihop", "Across all years, which SKU has the highest return rate (total quantity_returned divided by total quantity sold), and what is its most common return reason?",
        [ent(hs), ent(reason)], tools=["get_return_rate_analysis"])
    pr2 = pr.assign(rpb=pr.revenue_generated_inr / pr.budget_inr)
    b = pr2.loc[pr2.rpb.idxmax()]
    add("M05", "multihop", "Which promotion generated the most revenue per rupee of budget (revenue_generated_inr / budget_inr), and what was that ratio?",
        [ent(b["name"]), num(b.rpb, 0.05)], tools=["get_promotion_inventory_impact"])
    on = t25[t25.channel == "Online"].groupby("city").net_revenue_inr.sum()
    tc = on.idxmax(); share = 100 * on.max() / t25[t25.city == tc].net_revenue_inr.sum()
    add("M06", "multihop", "Which city had the highest Online-channel net revenue in 2025, and what share (%) of that city's total 2025 net revenue came from the Online channel?",
        [ent(tc), num(share, 0.5)], tools=["get_channel_revenue_attribution"])
    bm = t25.groupby("brand").gross_margin_inr.sum(); tb = bm.idxmax()
    bs = t25[t25.brand == tb].groupby("sku_id").quantity.sum()
    add("M07", "multihop", "Which brand earned the most total gross margin (sum of gross_margin_inr) in 2025, and which of that brand's SKUs sold the most units in 2025?",
        [ent(tb, ["HUFT"]), ent(bs.idxmax())], tools=["get_brand_performance"])
    t24s = t24.merge(s[["store_id", "region"]], on="store_id", how="left")
    g = (rg / t24s.groupby("region").net_revenue_inr.sum() - 1) * 100
    add("M08", "multihop", "Joining transactions to stores on store_id, which region had the highest percentage growth in net revenue from 2024 to 2025, and what was the growth percentage?",
        [ent(g.idxmax()), num(g.max(), 0.15)], tools=["get_franchise_inventory_comparison"])
    fr = sp25.groupby("supplier_name").fill_rate_pct.mean(); ws = fr.idxmin()
    add("M09", "multihop", "Which supplier had the lowest average fill_rate_pct across its 2025 monthly reviews, and what was its average lead_time_actual_days across those same 2025 reviews?",
        [ent(ws), num(sp25[sp25.supplier_name == ws].lead_time_actual_days.mean(), 0.15)],
        tools=["get_supplier_lead_time_tracker", "get_supplier_fill_rate_trend"])
    srev = t25.groupby("store_id").net_revenue_inr.sum(); ts = srev.idxmax()
    add("M10", "multihop", "Which store_id had the highest net revenue in 2025, and in which city is that store located (per the stores table)?",
        [ent(ts), ent(s.set_index("store_id").loc[ts, "city"])], tools=["get_store_level_demand_intelligence"])
    brf = r.groupby("brand").refund_inr.sum(); tbr = brf.idxmax()
    add("M11", "multihop", "Which brand has the highest total refund amount (sum of refund_inr across all returns), and how many SKUs of that brand are in the product catalog?",
        [ent(tbr, ["HUFT"]), num((p.brand == tbr).sum(), 0.01)], tools=["get_return_rate_analysis", "get_brand_performance"])
    u25 = t25.groupby("sku_id").quantity.sum(); tu = u25.idxmax()
    add("M12", "multihop", f"Which SKU sold the most units in 2025, and what was its inventory on the last date of the demand data ({last.date()}) according to the SKU-level daily demand table?",
        [ent(tu), num(cov.loc[tu, "inventory"], 0.5)], tools=["get_sku_360", "get_inventory_status"])
    cb = cc[cc.date.dt.year == 2025]
    br = cb[cb.temp_breach].sku_id.value_counts()
    # pick a multihop on cold chain with a unique answer if possible
    if len(br) and (br == br.max()).sum() == 1:
        add("M13", "multihop", "Among cold-chain SKUs, which one had the most temperature-breach records in 2025, and what is its catalog lead time in days?",
            [ent(br.idxmax()), num(p.set_index("sku_id").loc[br.idxmax(), "lead_time_days"], 0.01)],
            tools=["get_cold_chain_monitor"])
    else:
        rk = cc.groupby("sku_id").units_at_risk_of_expiry.sum(); tk = rk.idxmax()
        add("M13", "multihop", "Among cold-chain SKUs, which one has the largest total units_at_risk_of_expiry across the full cold-chain history, and what is its catalog lead time in days?",
            [ent(tk), num(p.set_index("sku_id").loc[tk, "lead_time_days"], 0.01)], tools=["get_cold_chain_monitor"])
    hl = c.groupby("segment").lifetime_value_inr.mean().idxmax()
    add("M14", "multihop", "For the customer segment with the highest average lifetime_value_inr, how many customers are in that segment and what is the average lifetime value (INR)?",
        [num((c.segment == hl).sum(), 0.01), num(ltv.max(), ltv.max() * 0.01)],
        tools=["get_customer_segmentation_insights"], note=f"segment {hl}")

    # ----------------------------------------------------------------- conflict
    # Stale system-prompt counts vs data.
    add("C01", "conflict", "How many stores does the company have in total (per the stores table)?",
        [num(len(s), 0.01)], distractors=[67, 85], fix=True,
        tools=["get_franchise_inventory_comparison", "get_store_inventory_breakdown"],
        note="prompt says 67; store_daily_inventory covers 85 stores")
    add("C02", "conflict", "How many SKUs are in the product catalog?", [num(len(p), 0.01)],
        distractors=[65], fix=True, tools=["get_supply_chain_dashboard", "get_abc_xyz_analysis"])
    add("C03", "conflict", "How many customers are in the customer table?", [num(len(c), 0.01)],
        distractors=[5000], fix=True, tools=["get_customer_segmentation_insights"])
    add("C04", "conflict", "How many sales transaction rows are in the sales transactions data?",
        [num(len(t), 0.01)], distractors=[50000], fix=True)
    add("C05", "conflict", "How many return records are in the returns data?", [num(len(r), 0.01)],
        distractors=[1500], fix=True, tools=["get_return_rate_analysis"])
    # Tool-vs-tool conflicts (stockout velocity).
    for qid, sk in [("C06", "EXT_059"), ("C07", "EXT_077"), ("C08", "FOOD_D011")]:
        cv = cov.loc[sk, "cover"]
        add(qid, "conflict",
            f"Using SKU {sk}'s average daily demand over the last 30 days and its current inventory in the SKU-level demand data, how many days of inventory cover does it have?",
            [num(cv, max(0.3, 0.07 * cv))], distractors=[round(old.loc[sk, "dtz"], 1)], fix=True,
            tools=["get_sku_360", "get_inventory_status", "get_stockout_risk"],
            note=f"original get_stockout_prediction (single-day velocity) says {old.loc[sk, 'dtz']:.1f} days")
    crit = cov[cov.cover < cov.lead_time_days]
    old_crit = int(((old.dtz < old["lead"]) & (old.vel > 0)).sum())
    add("C09", "conflict",
        "How many SKUs are critical right now, defining critical as: current inventory divided by the SKU's average daily demand over the last 30 days is less than its lead_time_days (SKU-level demand data, latest date)?",
        [num(len(crit), 0.01)], distractors=[old_crit], fix=True,
        tools=["get_supply_chain_dashboard", "get_stockout_risk", "get_reorder_list"],
        note=f"original get_stockout_prediction reports {old_crit} critical")
    lo = cov.sort_values("cover").iloc[0]
    add("C10", "conflict",
        "Which SKU has the fewest days of inventory cover, computed as current inventory divided by its average daily demand over the last 30 days (SKU-level demand data)?",
        [ent(lo.name)], distractors=[old.sort_values("dtz").index[0]], fix=True,
        tools=["get_stockout_risk", "get_inventory_status"],
        note=f"original get_stockout_prediction ranks {old.sort_values('dtz').index[0]} first")
    add("C11", "conflict", "Is SKU EXT_059 below its reorder point right now? Use reorder point = 30-day average daily demand x lead_time_days + 1.65 x std of daily demand over the last 90 days x sqrt(lead_time_days). Answer YES or NO and give the reorder point.",
        None, fix=True, tools=["get_sku_360", "get_reorder_list", "get_demand_forecast"])
    # fill C11 parts
    sk = "EXT_059"; h = dd[dd.sku_id == sk].sort_values("date")
    a30 = h[h.date > last - pd.Timedelta(days=30)].demand.mean(); s90 = h[h.date > last - pd.Timedelta(days=90)].demand.std()
    lt = float(p.set_index("sku_id").loc[sk, "lead_time_days"]); rop = a30 * lt + 1.65 * s90 * np.sqrt(lt)
    below = cov.loc[sk, "inventory"] < rop
    Q[-1]["parts"] = [ent("YES" if below else "NO"), num(rop, max(3, 0.03 * rop))]
    Q[-1]["distractors"] = ["YES"] if not below else ["NO"]
    tri = sp25[sp25.supplier_name == "Trixie India"].on_time_delivery_pct.mean()
    add("C12", "conflict", "What was Trixie India's average on-time delivery percentage across its 2025 monthly reviews only?",
        [num(tri, 0.1)], distractors=[round(sp[sp.supplier_name == "Trixie India"].on_time_delivery_pct.mean(), 1)],
        tools=["get_supplier_lead_time_tracker", "get_supplier_fill_rate_trend"],
        note="get_supplier_lead_time_tracker averages all 36 months")
    rf = sp25[sp25.supplier_name == "Ruffwear India"].on_time_delivery_pct.mean()
    add("C13", "conflict", "Which supplier had the worst average on-time delivery percentage in 2025, and what was that 2025 average? Then give the total current inventory (latest date, SKU-level demand data) across that supplier's SKUs.",
        [ent(otd.idxmin()), num(otd.min(), 0.05),
         num(dd[(dd.date == last) & (dd.supplier == otd.idxmin())].inventory.sum(), 1)],
        distractors=[round(sp[sp.supplier_name == otd.idxmin()].on_time_delivery_pct.mean(), 1)],
        tools=["get_supplier_lead_time_tracker", "get_supplier_fill_rate_trend"])
    inv_sku = cov.loc["EXT_059", "inventory"]; inv_store = old.loc["EXT_059", "inv"]
    add("C14", "conflict", "What is the current on-hand inventory of SKU EXT_059 according to the SKU-level daily demand table (latest date)?",
        [num(inv_sku, 0.5)], distractors=[inv_store],
        tools=["get_sku_360", "get_inventory_status"],
        note=f"store-level table sums to {inv_store} on its latest snapshot")
    # Tools whose (pre-fix) output carries the distractor value, used to tell
    # whether the agent was actually exposed to the conflict during a run.
    # "SYSTEM_PROMPT" = the stale count lives in the original system prompt.
    dt = {"C01": ["SYSTEM_PROMPT"], "C02": ["SYSTEM_PROMPT"], "C03": ["SYSTEM_PROMPT"],
          "C04": ["SYSTEM_PROMPT"], "C05": ["SYSTEM_PROMPT"],
          "C06": ["get_stockout_prediction"], "C07": ["get_stockout_prediction"],
          "C08": ["get_stockout_prediction"], "C09": ["get_stockout_prediction"],
          "C10": ["get_stockout_prediction"], "C11": ["get_stockout_prediction"],
          "C12": ["get_supplier_lead_time_tracker"], "C13": ["get_supplier_lead_time_tracker"],
          "C14": ["get_stockout_prediction"], "A05": ["get_supplier_lead_time_tracker"]}
    for q in Q:
        q["distractor_sources"] = dt.get(q["id"], [])
    return Q


def main():
    qs = build()
    with OUT.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    from collections import Counter
    print(len(qs), Counter(q["category"] for q in qs))
    for q in qs:
        print(q["id"], [pp["value"] for pp in q["parts"]], "distr:", q["distractors"], q["note"])


if __name__ == "__main__":
    main()
