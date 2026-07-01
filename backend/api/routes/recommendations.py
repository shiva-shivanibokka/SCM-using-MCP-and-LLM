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
    sku_orders = txn.groupby("sku_id")["order_id"].nunique().to_dict()

    pair_counts: dict[tuple[str, str], int] = {}
    for items in baskets:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                key = (items[i], items[j])
                pair_counts[key] = pair_counts.get(key, 0) + 1

    prod = load_products().drop_duplicates("sku_id").set_index("sku_id")
    names = prod["name"].to_dict()
    cats = prod["category"].to_dict()

    rows = []
    for (a, b), n in pair_counts.items():
        if n < 3:
            continue
        oa = sku_orders.get(a, 0) or 1
        ob = sku_orders.get(b, 0) or 1
        rows.append({
            "sku_a": a, "sku_a_name": names.get(a, a), "sku_a_category": cats.get(a, ""),
            "sku_b": b, "sku_b_name": names.get(b, b), "sku_b_category": cats.get(b, ""),
            "co_purchases": n,
            "support_pct": round(100.0 * n / total_orders, 3) if total_orders else 0.0,
            "lift": round(n * total_orders / (oa * ob), 2),
            "conf_a_to_b": round(100.0 * n / oa, 1),
            "conf_b_to_a": round(100.0 * n / ob, 1),
        })
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
    """Products most often bought alongside a SKU. Ranked by lift (association
    strength) when available, so genuine complements outrank merely-popular items;
    `also_bought_pct` is the % of this SKU's buyers who also bought the partner."""
    df = _co_purchase()
    if df.empty:
        return {"sku_id": sku_id, "recommendations": [], "ranked_by": "none"}

    has_lift = {"lift", "conf_a_to_b", "conf_b_to_a"}.issubset(df.columns)

    # A pair is symmetric: match sku_id on either side, return the other side.
    # The "also bought %" is the confidence FROM this SKU's side.
    left = df[df["sku_a"] == sku_id].rename(columns={
        "sku_b": "sku_id", "sku_b_name": "name", "sku_b_category": "category",
        **({"conf_a_to_b": "also_bought_pct"} if has_lift else {}),
    })
    right = df[df["sku_b"] == sku_id].rename(columns={
        "sku_a": "sku_id", "sku_a_name": "name", "sku_a_category": "category",
        **({"conf_b_to_a": "also_bought_pct"} if has_lift else {}),
    })
    cols = ["sku_id", "name", "category", "co_purchases", "support_pct"]
    if has_lift:
        cols += ["lift", "also_bought_pct"]
    combined = pd.concat([left[cols], right[cols]], ignore_index=True)

    if has_lift:
        # Only recommend genuine complements: bought together more than chance
        # (lift > 1.2) and often enough to trust (co_purchases >= 10). Ranked by
        # association strength. Empty is a valid, honest answer.
        pool = (
            combined[(combined["co_purchases"] >= 10) & (combined["lift"] > 1.2)]
            .sort_values("lift", ascending=False)
            .head(top_n)
        )
        ranked_by = "lift"
    else:
        pool = combined.sort_values("co_purchases", ascending=False).head(top_n)
        ranked_by = "co_purchases"
    return {"sku_id": sku_id, "recommendations": pool.to_dict("records"), "ranked_by": ranked_by}
