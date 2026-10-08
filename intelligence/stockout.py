"""Stockout prediction — velocity-based days-to-zero and reorder quantities.

Ported from the pre-revamp HUFT stockout engine, adapted to this project's data
model: a store_inventory snapshot carrying demand (daily velocity), inventory,
and lead_time_days per (store, SKU). Rolled up to per-SKU across stores.

Pure compute — callers pass the DataFrame, so the FastAPI backend and the MCP
agent tool share one implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lower number = more urgent (used for sorting).
RISK_RANK = {"critical": 0, "warning": 1, "watch": 2, "healthy": 3, "excess": 4, "dead": 5}


def _latest_snapshot(inv: pd.DataFrame) -> pd.DataFrame:
    """From a store_inventory frame (Postgres snapshot or full CSV history),
    keep the most recent row per (store, SKU)."""
    if "date" in inv.columns:
        inv = (inv.sort_values("date")
                  .groupby(["store_id", "sku_id"], as_index=False)
                  .tail(1))
    return inv


def _bucket(days_to_zero: float, velocity: float, inventory: float,
            lead_time: float, safety_days: int) -> str:
    if velocity <= 0:
        return "dead" if inventory > 0 else "healthy"
    if days_to_zero < lead_time:
        return "critical"
    if days_to_zero < lead_time + safety_days:
        return "warning"
    if days_to_zero < 30:
        return "watch"
    if days_to_zero <= 90:
        return "healthy"
    return "excess"


def _trailing_velocity(inv: pd.DataFrame, window_days: int) -> pd.Series | None:
    """Mean daily demand per (store, SKU) over the trailing `window_days`
    calendar days ending at the latest date in the frame. Returns None when the
    frame has no usable history (e.g. a single-date snapshot), in which case the
    caller falls back to the snapshot's own demand column."""
    if "date" not in inv.columns or window_days <= 1:
        return None
    dates = pd.to_datetime(inv["date"])
    if dates.nunique() < 2:
        return None
    cutoff = dates.max() - pd.Timedelta(days=window_days)
    recent = inv.loc[dates > cutoff]
    return recent.groupby(["store_id", "sku_id"])["demand"].mean()


def predict_stockouts(inv: pd.DataFrame, safety_stock_days: int = 7,
                      risk_filter: str | None = None,
                      velocity_window_days: int = 28) -> dict:
    """Per-SKU stockout risk. Returns {"rows": [...], "summary": {...}}.

    velocity   = sum across stores of each store's mean daily demand over the
                 trailing `velocity_window_days` (default 28). Previously this
                 was the latest single day's demand, so a one-day spike made
                 a SKU look critical. velocity_window_days=1 reproduces that.
    days_to_zero = current inventory / velocity
    reorder_qty  = max(0, (lead_time + safety_stock) * velocity - inventory)
    risk buckets = critical < lead_time < warning < +safety < watch < 30d < healthy < 90d < excess
    """
    if inv is None or inv.empty:
        return {"rows": [], "summary": {}}

    snap = _latest_snapshot(inv)
    vel = _trailing_velocity(inv, velocity_window_days)
    if vel is not None:
        snap = snap.drop(columns=["demand"]).merge(
            vel.rename("demand").reset_index(), on=["store_id", "sku_id"], how="left")
        snap["demand"] = snap["demand"].fillna(0.0)
    grp = snap.groupby("sku_id")
    agg = grp.agg(
        name=("name", "first"),
        category=("category", "first"),
        brand=("brand", "first"),
        inventory=("inventory", "sum"),
        daily_velocity=("demand", "sum"),
        lead_time_days=("lead_time_days", "max"),
        stores=("store_id", "nunique"),
    ).reset_index()

    rows = []
    for r in agg.itertuples(index=False):
        vel = float(r.daily_velocity)
        inv_units = float(r.inventory)
        lead = float(r.lead_time_days) if not pd.isna(r.lead_time_days) else 14.0
        dtz = inv_units / vel if vel > 0 else float("inf")
        reorder_point = (lead + safety_stock_days) * vel
        reorder_qty = max(0.0, reorder_point - inv_units)
        risk = _bucket(dtz, vel, inv_units, lead, safety_stock_days)
        rows.append({
            "sku_id": r.sku_id,
            "name": r.name,
            "category": r.category,
            "brand": r.brand,
            "stores": int(r.stores),
            "inventory": int(round(inv_units)),
            "daily_velocity": round(vel, 1),
            "days_to_zero": (None if dtz == float("inf") else round(dtz, 1)),
            "lead_time_days": int(lead),
            "reorder_qty": int(round(reorder_qty)),
            "risk": risk,
        })

    if risk_filter:
        rows = [x for x in rows if x["risk"] == risk_filter]

    rows.sort(key=lambda x: (RISK_RANK.get(x["risk"], 9),
                             x["days_to_zero"] if x["days_to_zero"] is not None else 1e9))

    summary = {b: 0 for b in RISK_RANK}
    for x in rows:
        summary[x["risk"]] += 1
    summary["total_skus"] = len(rows)
    summary["needs_reorder"] = sum(1 for x in rows if x["reorder_qty"] > 0
                                   and x["risk"] in ("critical", "warning"))
    return {"rows": rows, "summary": summary}
