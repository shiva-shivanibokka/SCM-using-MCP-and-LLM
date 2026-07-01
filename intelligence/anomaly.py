"""Anomaly detection — four rules-based detectors over sales and inventory.

Ported from the pre-revamp HUFT anomaly engine, adapted to this project's data:
  1. Sales crash        — SKU revenue drops >30% week-over-week
  2. Inventory spike     — inventory jumps >50% day-over-day (data-entry risk)
  3. Discount breach     — discount exceeds the per-channel ceiling
  4. Velocity-vs-stock   — high-velocity SKU with <14 days of cover

Pure compute: callers pass DataFrames, shared by the backend route and MCP tool.
"""
from __future__ import annotations

import pandas as pd

SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}

# Per-channel discount ceilings (%). Above this is flagged as a pricing breach.
DISCOUNT_CEILING = {"Online": 45.0, "App": 45.0, "Offline": 35.0}
# Ignore tiny SKUs so a crash from ₹200→₹100 doesn't spam the feed.
MATERIAL_REVENUE = 20000.0


def _detect_sales_crash(txn: pd.DataFrame) -> list[dict]:
    if txn.empty or "date" not in txn.columns:
        return []
    txn = txn.copy()
    txn["date"] = pd.to_datetime(txn["date"])
    end = txn["date"].max()
    cur = txn[(txn["date"] > end - pd.Timedelta(days=7)) & (txn["date"] <= end)]
    prev = txn[(txn["date"] > end - pd.Timedelta(days=14)) & (txn["date"] <= end - pd.Timedelta(days=7))]
    cur_rev = cur.groupby("sku_id")["net_revenue_inr"].sum()
    prev_rev = prev.groupby("sku_id")["net_revenue_inr"].sum()
    names = txn.drop_duplicates("sku_id").set_index("sku_id")["category"].to_dict()
    out = []
    for sku, p in prev_rev.items():
        if p < MATERIAL_REVENUE:
            continue
        c = float(cur_rev.get(sku, 0.0))
        drop = (p - c) / p
        if drop > 0.30:
            out.append({
                "type": "Sales crash",
                "severity": "critical" if drop > 0.6 else "warning",
                "entity": sku,
                "detail": f"Weekly revenue fell {drop * 100:.0f}% (₹{p:,.0f} → ₹{c:,.0f}) in {names.get(sku, '')}.",
                "value": round(drop * 100, 1),
            })
    return out


def _detect_inventory_spike(inv: pd.DataFrame) -> list[dict]:
    if inv.empty or "date" not in inv.columns:
        return []
    inv = inv.copy()
    inv["date"] = pd.to_datetime(inv["date"])
    dates = sorted(inv["date"].unique())
    if len(dates) < 2:
        return []
    last, prior = dates[-1], dates[-2]
    a = inv[inv["date"] == last].set_index(["store_id", "sku_id"])["inventory"]
    b = inv[inv["date"] == prior].set_index(["store_id", "sku_id"])["inventory"]
    joined = a.to_frame("now").join(b.to_frame("was"), how="inner")
    joined = joined[(joined["was"] >= 20) & (joined["now"] > joined["was"] * 1.5)]
    out = []
    for (store, sku), r in joined.iterrows():
        jump = (r["now"] - r["was"]) / r["was"]
        out.append({
            "type": "Inventory spike",
            "severity": "warning" if jump < 2 else "critical",
            "entity": f"{sku} @ {store}",
            "detail": f"Inventory jumped {jump * 100:.0f}% ({int(r['was'])} → {int(r['now'])}) overnight — possible data-entry error.",
            "value": round(jump * 100, 1),
        })
    return out[:25]


def _detect_discount_breach(txn: pd.DataFrame) -> list[dict]:
    if txn.empty or "discount_pct" not in txn.columns:
        return []
    out = []
    for channel, ceiling in DISCOUNT_CEILING.items():
        sub = txn[(txn["channel"] == channel) & (txn["discount_pct"] > ceiling)]
        if sub.empty:
            continue
        by_sku = sub.groupby("sku_id").agg(
            max_disc=("discount_pct", "max"), n=("txn_id", "count")
        ).sort_values("max_disc", ascending=False).head(10)
        for sku, r in by_sku.iterrows():
            out.append({
                "type": "Discount breach",
                "severity": "warning",
                "entity": f"{sku} ({channel})",
                "detail": f"{int(r['n'])} sales at up to {r['max_disc']:.0f}% discount — above the {ceiling:.0f}% {channel} ceiling.",
                "value": round(float(r["max_disc"]), 1),
            })
    return out


def _detect_velocity_risk(inv: pd.DataFrame) -> list[dict]:
    if inv.empty:
        return []
    if "date" in inv.columns:
        inv = inv.sort_values("date").groupby(["store_id", "sku_id"], as_index=False).tail(1)
    agg = inv.groupby("sku_id").agg(
        name=("name", "first"), inventory=("inventory", "sum"), velocity=("demand", "sum")
    ).reset_index()
    agg = agg[agg["velocity"] > 0]
    agg["cover"] = agg["inventory"] / agg["velocity"]
    risky = agg[agg["cover"] < 14].sort_values("cover").head(15)
    out = []
    for r in risky.itertuples(index=False):
        out.append({
            "type": "Velocity-vs-stock risk",
            "severity": "critical" if r.cover < 7 else "warning",
            "entity": r.sku_id,
            "detail": f"{r.name}: only {r.cover:.1f} days of cover at {r.velocity:.0f} units/day.",
            "value": round(float(r.cover), 1),
        })
    return out


def run_anomaly_detection(txn: pd.DataFrame, inv: pd.DataFrame) -> dict:
    """Run all four detectors. Returns {"anomalies": [...], "summary": {...}}."""
    anomalies = (
        _detect_sales_crash(txn) + _detect_inventory_spike(inv)
        + _detect_discount_breach(txn) + _detect_velocity_risk(inv)
    )
    anomalies.sort(key=lambda a: (SEVERITY_RANK.get(a["severity"], 9), -abs(a.get("value", 0))))
    summary = {"total": len(anomalies),
               "critical": sum(1 for a in anomalies if a["severity"] == "critical"),
               "warning": sum(1 for a in anomalies if a["severity"] == "warning")}
    by_type: dict[str, int] = {}
    for a in anomalies:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    summary["by_type"] = by_type
    return {"anomalies": anomalies, "summary": summary}
