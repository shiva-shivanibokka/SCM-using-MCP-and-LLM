from __future__ import annotations
from fastapi import APIRouter, Query

from backend.data_access import load_demand, load_store_inventory, load_stores

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


def _bucket(rows: list[dict]) -> dict:
    reorder, risk, dead, over = [], [], [], []
    for r in rows:
        cover = r["days_of_cover"]
        inv, dmd = r["inventory"], r["demand"]
        if cover < 7:
            risk.append(r)
        if cover < 14:
            reorder.append(r)
        if dmd == 0 and inv > 0:
            dead.append(r)
        if cover > 90:
            over.append(r)
    return {
        "reorder_list": reorder[:200],
        "stockout_risk": risk[:200],
        "dead_stock": dead[:200],
        "overstock": over[:200],
    }


@router.get("/stores")
def store_options():
    """Lightweight list for the store selector."""
    df = load_stores()
    out = [{"store_id": r["store_id"], "label": f'{r["store_id"]} · {r.get("city","")}'}
           for _, r in df.iterrows()]
    return {"stores": out}


@router.get("/health")
def health(store_id: str | None = Query(None)):
    if store_id:
        sdi = load_store_inventory()
        sub = sdi[sdi["store_id"] == store_id]
        latest = sub.sort_values("date").groupby("sku_id").tail(1)
        rows = []
        for _, r in latest.iterrows():
            inv, dmd = float(r["inventory"]), float(r["demand"])
            cover = float(r["days_of_supply"]) if r["days_of_supply"] == r["days_of_supply"] else (
                inv / dmd if dmd > 0 else 999.0)
            rows.append({"sku_id": r["sku_id"], "name": str(r.get("name", "")),
                         "inventory": inv, "demand": dmd, "days_of_cover": round(cover, 1)})
        return {"scope": store_id, **_bucket(rows)}

    # National view (all stores) from the aggregated demand table
    demand = load_demand()
    latest = demand.sort_values("date").groupby("sku_id").tail(1)
    rows = []
    for _, r in latest.iterrows():
        inv, dmd = float(r["inventory"]), float(r["demand"])
        cover = inv / dmd if dmd > 0 else 999.0
        rows.append({"sku_id": r["sku_id"], "name": str(r.get("name", "")),
                     "inventory": inv, "demand": dmd, "days_of_cover": round(cover, 1)})
    return {"scope": "all", **_bucket(rows)}
