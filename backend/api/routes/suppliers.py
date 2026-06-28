from __future__ import annotations
from fastapi import APIRouter

from backend.data_access import load_suppliers

router = APIRouter(prefix="/api/suppliers", tags=["suppliers"])

_METRICS = ["on_time_delivery_pct", "defect_rate_pct", "fill_rate_pct",
            "lead_time_actual_days", "quality_rating"]


@router.get("/scorecard")
def scorecard():
    df = load_suppliers()
    avail = [m for m in _METRICS if m in df.columns]
    agg = (df.groupby("supplier_name")[avail].mean().round(2)
           .reset_index().sort_values("on_time_delivery_pct", ascending=False)
           if "on_time_delivery_pct" in avail
           else df.groupby("supplier_name")[avail].mean().round(2).reset_index())
    return {"suppliers": agg.to_dict(orient="records")}
