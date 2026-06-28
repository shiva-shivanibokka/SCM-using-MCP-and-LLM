from __future__ import annotations
from fastapi import APIRouter

from backend.data_access import load_customers, load_returns

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


@router.get("/overview")
def overview():
    cust = load_customers()
    seg_col = _col(cust, "segment", "customer_segment")
    segments = cust[seg_col].value_counts().to_dict() if seg_col else {}

    ltv_by_segment = {}
    if seg_col and "lifetime_value_inr" in cust.columns:
        ltv = cust.groupby(seg_col)["lifetime_value_inr"].mean().round(0)
        ltv_by_segment = {str(k): float(v) for k, v in ltv.items()}

    ret = load_returns()
    cat_col = _col(ret, "category", "product_category")
    ret_by_cat = ret[cat_col].value_counts().to_dict() if cat_col else {}

    return {
        "segments": {str(k): int(v) for k, v in segments.items()},
        "ltv_by_segment": ltv_by_segment,
        "return_rate_by_category": {str(k): int(v) for k, v in ret_by_cat.items()},
        "brand_performance": [],
    }
