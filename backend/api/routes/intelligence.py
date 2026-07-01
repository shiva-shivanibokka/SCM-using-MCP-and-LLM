"""Operational intelligence endpoints — stockout, anomaly, and what-if.

Ported from the pre-revamp HUFT intelligence tabs, adapted to this project's
data model. Compute lives in the top-level `intelligence` package so the MCP
agent tools share the exact same logic.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import settings
from backend.data_access import load_products, load_store_inventory, load_transactions
from intelligence.anomaly import run_anomaly_detection
from intelligence.sql import run_query
from intelligence.stockout import predict_stockouts
from intelligence.whatif import simulate_discount_impact, simulate_restock_impact

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])


class SqlBody(BaseModel):
    sql: str


@router.post("/sql")
def ask_data_sql(body: SqlBody):
    """Run a guarded read-only SQL SELECT against the data (DuckDB). Returns
    {columns, rows, total, truncated, error}."""
    return run_query(body.sql, settings.DATA_DIR, max_rows=100)


class AskBody(BaseModel):
    question: str
    provider: str = "groq"
    model: str | None = None
    api_key: str | None = None


@router.post("/ask")
async def ask_data_nl(body: AskBody):
    """Natural-language → SQL. An LLM (BYOK, same providers as the chat) turns the
    question into a single SQL SELECT, which is then guarded and executed. Returns
    the same shape as /sql plus the generated `sql` so the UI can show/edit it."""
    import os

    from agent.agent import _call_llm, PROVIDER_ENV_KEYS, PROVIDERS
    from intelligence.sql import SCHEMA_TEXT, extract_sql

    provider = body.provider
    model = body.model or PROVIDERS.get(provider, {}).get("default_model")
    key = body.api_key or os.getenv(PROVIDER_ENV_KEYS.get(provider, ""), "")
    empty = {"columns": [], "rows": [], "total": 0, "truncated": False, "sql": None}
    if not key:
        return {**empty, "error": f"No {provider} API key — enter one in the model bar."}

    prompt = (
        "You translate a business question into exactly ONE DuckDB SQL query.\n"
        "Rules: a single read-only SELECT (or WITH … SELECT); DuckDB syntax; use only "
        "the tables/columns in the schema; no writes, DDL, comments, or explanation. "
        "Output ONLY the SQL.\n\n"
        f"Schema:\n{SCHEMA_TEXT}\n\nQuestion: {body.question}\n\nSQL:"
    )
    try:
        res = await _call_llm([{"role": "user", "content": prompt}], [], provider, model, key)
    except Exception as e:  # bad key, rate limit, etc.
        return {**empty, "error": f"LLM error: {e}"}

    sql = extract_sql(res.get("text") or "")
    if not sql:
        return {**empty, "error": "Could not turn that into SQL — try rephrasing.",
                "sql": (res.get("text") or "").strip()[:400]}
    result = run_query(sql, settings.DATA_DIR, max_rows=100)
    result["sql"] = sql
    return result


@router.get("/stockout")
def stockout(safety_stock_days: int = 7, risk: str | None = None, top_n: int = 100):
    """Per-SKU stockout risk: days-to-zero, reorder quantity, and risk bucket."""
    result = predict_stockouts(
        load_store_inventory(), safety_stock_days=safety_stock_days, risk_filter=risk
    )
    result["rows"] = result["rows"][:top_n]
    return result


@router.get("/anomaly")
def anomaly():
    """Four-detector anomaly scan: sales crash, inventory spike, discount breach,
    velocity-vs-stock risk."""
    return run_anomaly_detection(load_transactions(), load_store_inventory())


@router.get("/options")
def options():
    """Dropdown options for the what-if simulator: product categories and SKUs."""
    p = load_products()
    categories = sorted(x for x in p["category"].dropna().unique().tolist())
    skus = (
        p[["sku_id", "name"]]
        .drop_duplicates("sku_id")
        .sort_values("name")
        .to_dict("records")
    )
    return {"categories": categories, "skus": skus}


@router.get("/whatif/discount")
def whatif_discount(new_discount_pct: float = 20.0, category: str | None = None):
    """Project the impact of a new discount level on units, GMV, and net revenue."""
    return simulate_discount_impact(load_transactions(), category=category,
                                    new_discount_pct=new_discount_pct)


@router.get("/whatif/restock")
def whatif_restock(sku_id: str, restock_units: int = 500):
    """Project days-of-cover, overstock risk, GMV unlock, and ROI for a restock."""
    return simulate_restock_impact(load_store_inventory(), load_products(),
                                   sku_id=sku_id, restock_units=restock_units)
