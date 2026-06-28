from __future__ import annotations
from fastapi import APIRouter

from backend.data_access import load_transactions, load_demand, load_stores

router = APIRouter(prefix="/api/executive", tags=["executive"])


@router.get("/kpis")
def kpis():
    txn = load_transactions()
    revenue = float(txn["net_revenue_inr"].sum())
    margin = float(txn["gross_margin_inr"].sum())
    gross_margin_pct = round(margin / revenue * 100, 1) if revenue else 0.0

    # Revenue by region: join transactions -> stores on store_id
    stores = load_stores()[["store_id", "region"]]
    merged = txn.merge(stores, on="store_id", how="left")
    by_region = merged.groupby("region")["net_revenue_inr"].sum()

    by_sku = txn.groupby("sku_id")["net_revenue_inr"].sum().sort_values(ascending=False)

    # Inventory value from latest daily snapshot (inventory units * unit cost)
    demand = load_demand()
    latest = demand.sort_values("date").groupby("sku_id").tail(1)
    inventory_value = float((latest["inventory"] * latest["cost_inr"]).sum())
    stockout_rate = round(float((demand["demand"] == 0).mean() * 100), 1)

    return {
        "revenue": revenue,
        "gross_margin_pct": gross_margin_pct,
        "inventory_value": inventory_value,
        "stockout_rate": stockout_rate,
        "fill_rate": round(100.0 - stockout_rate, 1),
        "top_skus": [{"sku_id": k, "revenue": float(v)} for k, v in by_sku.head(10).items()],
        "bottom_skus": [{"sku_id": k, "revenue": float(v)} for k, v in by_sku.tail(10).items()],
        "revenue_by_region": {str(k): float(v) for k, v in by_region.items()},
    }
