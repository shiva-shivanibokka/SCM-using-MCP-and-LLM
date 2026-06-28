from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query

from backend.data_access import sku_history, load_products
from backend.forecasting.ensemble import ensemble_forecast

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.get("/skus")
def list_skus():
    df = load_products()
    name_col = "product_name" if "product_name" in df.columns else (
        "name" if "name" in df.columns else df.columns[1])
    return [{"sku_id": r["sku_id"], "name": str(r[name_col])}
            for _, r in df.iterrows()]


@router.get("/{sku_id}")
def forecast(sku_id: str,
             horizon: int = Query(30, ge=1, le=180),
             components: bool = False):
    history = sku_history(sku_id)
    if not history:
        raise HTTPException(404, f"unknown sku_id: {sku_id}")
    out = ensemble_forecast(history, horizon=horizon, components=components)
    return {"sku_id": sku_id, "horizon": horizon, **out}
