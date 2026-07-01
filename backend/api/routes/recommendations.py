"""Product recommendation endpoints.

Serves the "frequently bought together" signal from the co_purchase_pairs dbt
mart when it's built (Postgres or local DuckDB), and otherwise computes it on the
fly from the raw transactions' real multi-item baskets (order_id). Either way the
API shape is identical, so the frontend never cares where the numbers came from.
"""
from __future__ import annotations

import functools

import pandas as pd
from fastapi import APIRouter

from backend.data_access import load_mart, load_products, load_transactions

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@functools.lru_cache(maxsize=1)
def _co_purchase() -> pd.DataFrame:
    """Co-purchase pairs as a DataFrame. Prefers the dbt mart; falls back to a
    pandas computation over real order_id baskets. Cached (data is static)."""
    mart = load_mart("co_purchase_pairs")
    if mart is not None and not mart.empty:
        return mart.sort_values("co_purchases", ascending=False).reset_index(drop=True)

    # Fallback: compute from raw transactions grouped by order.
    txn = load_transactions()
    if txn.empty or "order_id" not in txn.columns:
        return pd.DataFrame(
            columns=["sku_a", "sku_a_name", "sku_a_category", "sku_b",
                     "sku_b_name", "sku_b_category", "co_purchases", "support_pct"]
        )

    baskets = txn.groupby("order_id")["sku_id"].apply(lambda s: sorted(set(s)))
    baskets = baskets[baskets.apply(len) >= 2]
    total_orders = int(txn["order_id"].nunique())

    pair_counts: dict[tuple[str, str], int] = {}
    for items in baskets:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                key = (items[i], items[j])
                pair_counts[key] = pair_counts.get(key, 0) + 1

    prod = load_products().drop_duplicates("sku_id").set_index("sku_id")
    names = prod["name"].to_dict()
    cats = prod["category"].to_dict()

    rows = [
        {
            "sku_a": a, "sku_a_name": names.get(a, a), "sku_a_category": cats.get(a, ""),
            "sku_b": b, "sku_b_name": names.get(b, b), "sku_b_category": cats.get(b, ""),
            "co_purchases": n,
            "support_pct": round(100.0 * n / total_orders, 3) if total_orders else 0.0,
        }
        for (a, b), n in pair_counts.items()
        if n >= 3
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("co_purchases", ascending=False).reset_index(drop=True)


@router.get("/overview")
def overview(top_n: int = 20):
    """Top co-purchased product pairs plus headline KPIs."""
    df = _co_purchase()
    pairs = df.head(top_n).to_dict("records")
    top = pairs[0] if pairs else None
    return {
        "pairs": pairs,
        "total_pairs": int(len(df)),
        "top_pair": (
            f'{top["sku_a_name"]} + {top["sku_b_name"]}' if top else None
        ),
        "max_co_purchases": int(df["co_purchases"].max()) if not df.empty else 0,
        "source": "dbt_mart" if load_mart("co_purchase_pairs") is not None else "computed",
    }


@router.get("/for-sku/{sku_id}")
def for_sku(sku_id: str, top_n: int = 8):
    """Given a SKU, the products most often bought alongside it."""
    df = _co_purchase()
    if df.empty:
        return {"sku_id": sku_id, "recommendations": []}

    # A pair is symmetric: match sku_id on either side, return the other side.
    left = df[df["sku_a"] == sku_id].rename(
        columns={"sku_b": "sku_id", "sku_b_name": "name", "sku_b_category": "category"}
    )
    right = df[df["sku_b"] == sku_id].rename(
        columns={"sku_a": "sku_id", "sku_a_name": "name", "sku_a_category": "category"}
    )
    cols = ["sku_id", "name", "category", "co_purchases", "support_pct"]
    combined = (
        pd.concat([left[cols], right[cols]], ignore_index=True)
        .sort_values("co_purchases", ascending=False)
        .head(top_n)
    )
    return {"sku_id": sku_id, "recommendations": combined.to_dict("records")}
