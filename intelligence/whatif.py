"""What-if simulator — discount-elasticity and restock-ROI scenarios.

Ported from the pre-revamp HUFT what-if engine, adapted to this project's data.
  - Discount scenario: estimate price elasticity from history, then project how a
    new discount changes units, GMV, and net revenue.
  - Restock scenario: project new days-of-cover, overstock risk, GMV unlocked,
    restock cost, and ROI for adding N units of a SKU.

GMV = MRP value (unit_price × qty); NR = net revenue after discount.
Pure compute — shared by the backend route and the MCP tool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _estimate_elasticity(sub: pd.DataFrame) -> float:
    """Rough price elasticity of demand from discount variation in history.
    Compares AVERAGE DAILY units in low- vs high-discount days (normalising out the
    fact that promo periods simply span more days). Clamped to a realistic retail
    range [-2.5, -0.5] and defaulted to -1.2 when the data is too thin to fit."""
    default = -1.2
    if sub.empty or "discount_pct" not in sub.columns or "date" not in sub.columns:
        return default
    med = sub["discount_pct"].median()
    low = sub[sub["discount_pct"] <= med]
    high = sub[sub["discount_pct"] > med]
    if low.empty or high.empty:
        return default
    days_low = max(1, low["date"].nunique())
    days_high = max(1, high["date"].nunique())
    price_low = (low["unit_price_inr"] * (1 - low["discount_pct"] / 100)).mean()
    price_high = (high["unit_price_inr"] * (1 - high["discount_pct"] / 100)).mean()
    # Average units PER DAY at each discount level (confound-corrected).
    u_low = low["quantity"].sum() / days_low
    u_high = high["quantity"].sum() / days_high
    if price_low <= 0 or price_high <= 0 or u_low <= 0 or u_high <= 0:
        return default
    denom = np.log(price_high / price_low)
    if abs(denom) < 1e-6:
        return default
    elasticity = np.log(u_high / u_low) / denom
    if not np.isfinite(elasticity):
        return default
    # Clamp to a conservative retail range. (Synthetic promos are timed to demand
    # peaks, which inflates the raw fit, so we cap it to keep projections sane.)
    return float(np.clip(elasticity, -1.8, -0.5))


def simulate_discount_impact(txn: pd.DataFrame, category: str | None = None,
                             new_discount_pct: float = 20.0) -> dict:
    sub = txn
    if category:
        sub = sub[sub["category"].str.contains(category, case=False, na=False)]
    if sub is None or sub.empty:
        return {"error": "No transactions match that filter."}

    gmv = float((sub["unit_price_inr"] * sub["quantity"]).sum())
    nr = float(sub["net_revenue_inr"].sum())
    units = float(sub["quantity"].sum())
    base_disc = (1 - nr / gmv) * 100 if gmv > 0 else 0.0

    elasticity = _estimate_elasticity(sub)
    price_ratio = (1 - new_discount_pct / 100) / max(1e-6, (1 - base_disc / 100))
    units_ratio = price_ratio ** elasticity
    proj_units = units * units_ratio
    proj_gmv = gmv * units_ratio
    proj_nr = proj_gmv * (1 - new_discount_pct / 100)

    return {
        "scenario": "discount",
        "category": category or "All categories",
        "elasticity": round(elasticity, 2),
        "baseline": {
            "avg_discount_pct": round(base_disc, 1),
            "units": int(units), "gmv": round(gmv), "net_revenue": round(nr),
        },
        "projected": {
            "discount_pct": round(new_discount_pct, 1),
            "units": int(proj_units), "gmv": round(proj_gmv), "net_revenue": round(proj_nr),
        },
        "delta": {
            "units_pct": round((units_ratio - 1) * 100, 1),
            "net_revenue": round(proj_nr - nr),
            "net_revenue_pct": round((proj_nr / nr - 1) * 100, 1) if nr > 0 else 0.0,
        },
        "note": "Elasticity estimated from historical discount variation; a rough planning guide, not a guarantee.",
    }


def simulate_restock_impact(inv: pd.DataFrame, products: pd.DataFrame,
                            sku_id: str, restock_units: int) -> dict:
    if "date" in inv.columns:
        inv = inv.sort_values("date").groupby(["store_id", "sku_id"], as_index=False).tail(1)
    s = inv[inv["sku_id"] == sku_id]
    if s.empty:
        return {"error": f"SKU {sku_id} not found in inventory."}
    inventory = float(s["inventory"].sum())
    velocity = float(s["demand"].sum())

    prow = products[products["sku_id"] == sku_id]
    price = float(prow["price_inr"].iloc[0]) if not prow.empty else float(s["price_inr"].iloc[0])
    cost = float(prow["cost_inr"].iloc[0]) if not prow.empty else float(s["cost_inr"].iloc[0])
    name = prow["name"].iloc[0] if not prow.empty and "name" in prow else s.get("name", pd.Series([sku_id])).iloc[0]

    cur_days = inventory / velocity if velocity > 0 else float("inf")
    new_inv = inventory + restock_units
    new_days = new_inv / velocity if velocity > 0 else float("inf")
    restock_cost = restock_units * cost
    profit = restock_units * (price - cost)
    roi_pct = (profit / restock_cost * 100) if restock_cost > 0 else 0.0
    gmv_unlock = restock_units * price
    overstock = (new_days != float("inf")) and (new_days > 90)

    return {
        "scenario": "restock",
        "sku_id": sku_id, "name": name,
        "restock_units": int(restock_units),
        "current": {
            "inventory": int(inventory), "daily_velocity": round(velocity, 1),
            "days_of_cover": (None if cur_days == float("inf") else round(cur_days, 1)),
        },
        "projected": {
            "inventory": int(new_inv),
            "days_of_cover": (None if new_days == float("inf") else round(new_days, 1)),
            "overstock_risk": overstock,
        },
        "economics": {
            "restock_cost": round(restock_cost), "gmv_unlock": round(gmv_unlock),
            "gross_profit": round(profit), "roi_pct": round(roi_pct, 1),
        },
        "note": "Assumes restocked units sell through at current price/velocity. Overstock flagged above 90 days cover.",
    }
