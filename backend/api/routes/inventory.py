from __future__ import annotations
from fastapi import APIRouter

from backend.data_access import load_demand

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


@router.get("/health")
def health():
    demand = load_demand()
    latest = demand.sort_values("date").groupby("sku_id").tail(1)

    reorder, risk, dead, over, cold = [], [], [], [], []
    for _, r in latest.iterrows():
        sku = r["sku_id"]
        inv = float(r["inventory"])
        dmd = float(r["demand"])
        cover = inv / dmd if dmd > 0 else 999.0
        rec = {"sku_id": sku, "name": str(r.get("name", "")),
               "inventory": inv, "days_of_cover": round(cover, 1)}
        if cover < 7:
            risk.append(rec)
        if cover < 14:
            reorder.append(rec)
        if dmd == 0 and inv > 0:
            dead.append(rec)
        if cover > 90:
            over.append(rec)
        if bool(r.get("is_cold_chain", False)):
            cold.append(rec)

    return {
        "reorder_list": reorder[:50],
        "stockout_risk": risk[:50],
        "dead_stock": dead[:50],
        "overstock": over[:50],
        "cold_chain_status": cold[:50],
    }
