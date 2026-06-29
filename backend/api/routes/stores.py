from __future__ import annotations
from fastapi import APIRouter, HTTPException

from backend.data_access import (
    load_stores, load_transactions, load_store_inventory, load_mart,
)

router = APIRouter(prefix="/api/stores", tags=["stores"])


@router.get("/grid")
def grid():
    df = load_stores()
    return {"stores": df.to_dict(orient="records")}


@router.get("/{store_id}")
def detail(store_id: str):
    stores = load_stores()
    row = stores[stores["store_id"] == store_id]
    if row.empty:
        raise HTTPException(404, f"unknown store_id: {store_id}")
    profile = row.iloc[0].to_dict()

    # Sales KPIs from transactions for this store
    txn = load_transactions()
    t = txn[txn["store_id"] == store_id]
    revenue = float(t["net_revenue_inr"].sum())
    units = int(t["quantity"].sum()) if "quantity" in t.columns else 0
    orders = int(len(t))
    margin = float(t["gross_margin_inr"].sum())
    margin_pct = round(margin / revenue * 100, 1) if revenue else 0.0
    top_categories = [  # noqa: E501 (kept from raw — the mart doesn't carry category breakdown)
        {"category": str(k), "revenue": float(v)}
        for k, v in t.groupby("category")["net_revenue_inr"].sum()
        .sort_values(ascending=False).head(10).items()
    ]
    channel_mix = {str(k): int(v) for k, v in t["channel"].value_counts().items()} \
        if "channel" in t.columns else {}

    # Inventory health from store_daily_inventory (latest per SKU at this store)
    sdi = load_store_inventory()
    s = sdi[sdi["store_id"] == store_id]
    latest = s.sort_values("date").groupby("sku_id").tail(1)
    inventory_value = float((latest["inventory"] * latest["cost_inr"]).sum())
    sku_count = int(latest["sku_id"].nunique())
    risk_counts = {str(k): int(v) for k, v in latest["risk_status"].value_counts().items()} \
        if "risk_status" in latest.columns else {}

    # SKU lists per risk status, so the UI can show "all SKUs at CRITICAL" etc.
    risk_skus = {"OK": [], "WARNING": [], "CRITICAL": []}
    if "risk_status" in latest.columns:
        for _, r in latest.sort_values("days_of_supply").iterrows():
            status = str(r["risk_status"])
            if status in risk_skus:
                risk_skus[status].append({
                    "sku_id": str(r["sku_id"]),
                    "name": str(r.get("name", "")),
                    "inventory": float(r["inventory"]),
                    "days_of_cover": round(float(r["days_of_supply"]), 1)
                    if r["days_of_supply"] == r["days_of_supply"] else None,
                })

    # Prefer the dbt `store_kpis` mart for headline numbers; fall back to the
    # raw pandas compute above if dbt hasn't been built. (sku_count comes from
    # the live inventory snapshot either way.)
    kpis = {
        "revenue": revenue,
        "orders": orders,
        "units": units,
        "gross_margin_pct": margin_pct,
        "inventory_value": inventory_value,
        "sku_count": sku_count,
    }
    kpis_source = "raw"
    mart = load_mart("store_kpis")
    if mart is not None:
        m = mart[mart["store_id"] == store_id]
        if not m.empty:
            r = m.iloc[0]
            kpis.update({
                "revenue": float(r["revenue"]),
                "orders": int(r["orders"]),
                "units": int(r["units"]),
                "gross_margin_pct": float(r["gross_margin_pct"]),
                "inventory_value": float(r["inventory_value"]),
            })
            kpis_source = "dbt:store_kpis"

    return {
        "profile": profile,
        "kpis": kpis,
        "kpis_source": kpis_source,
        "top_categories": top_categories,
        "channel_mix": channel_mix,
        "risk_counts": risk_counts,
        "risk_skus": risk_skus,
    }
