"""Guarded read-only SQL over the data (DuckDB) — one shared implementation.

Used by both the FastAPI "Ask Your Data" endpoint and the agent's run_sql_query
MCP tool, so the safety rules live in exactly one place. Only a single read-only
SELECT/WITH is permitted; writes, DDL, multi-statement, and DuckDB file/system
functions are blocked, and results are row-capped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Friendly view name → source CSV in the data dir.
VIEWS = {
    "products": "huft_products.csv",
    "stores": "huft_stores.csv",
    "transactions": "huft_sales_transactions.csv",
    "demand": "huft_daily_demand.csv",
    "customers": "huft_customers.csv",
    "promotions": "huft_promotions.csv",
    "returns": "huft_returns.csv",
    "suppliers": "huft_supplier_performance.csv",
    "store_inventory": "store_daily_inventory.csv",
}

# Keywords use word boundaries (so "created_at" is fine); dangerous DuckDB
# functions use substring matching (so read_csv_auto / parquet_scan / glob can't
# slip past a word boundary).
_FORBIDDEN_KW = (
    "insert", "update", "delete", "drop", "create", "alter", "attach", "detach",
    "copy", "pragma", "export", "call", "merge", "truncate", "grant", "replace",
)
_FORBIDDEN_SUB = (
    "read_", "_scan", "glob", "sniff", "system", "getenv", "install", "load",
)

_CON_CACHE: dict[str, object] = {}


def is_safe_select(sql: str) -> bool:
    s = sql.strip().rstrip(";").lower()
    if not (s.startswith("select") or s.startswith("with")):
        return False
    if ";" in s:  # single statement only
        return False
    if any(re.search(rf"\b{kw}\b", s) for kw in _FORBIDDEN_KW):
        return False
    return not any(sub in s for sub in _FORBIDDEN_SUB)


def _get_con(data_dir):
    key = str(data_dir)
    if key in _CON_CACHE:
        return _CON_CACHE[key]
    import duckdb

    con = duckdb.connect(database=":memory:")
    for name, csv in VIEWS.items():
        p = Path(data_dir) / csv
        if p.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{p.as_posix()}')")
    _CON_CACHE[key] = con
    return con


def run_query(sql: str, data_dir, max_rows: int = 100) -> dict:
    """Execute a guarded SELECT. Returns
    {columns: [...], rows: [ {col: val} ], total: int, truncated: bool, error: str|None}.
    """
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return {"columns": [], "rows": [], "total": 0, "truncated": False,
                "error": "SQL querying is unavailable here (duckdb not installed)."}
    if not is_safe_select(sql):
        return {"columns": [], "rows": [], "total": 0, "truncated": False,
                "error": "Only a single read-only SELECT (or WITH … SELECT) is allowed — "
                         "no writes, DDL, or file access."}
    try:
        con = _get_con(data_dir)
        df = con.execute(sql).fetchdf()
        total = int(len(df))
        head = df.head(max_rows)
        # json round-trip makes numpy/NaN/timestamps JSON-safe.
        rows = json.loads(head.to_json(orient="records", date_format="iso"))
        return {"columns": list(head.columns), "rows": rows, "total": total,
                "truncated": total > len(head), "error": None}
    except Exception as e:
        return {"columns": [], "rows": [], "total": 0, "truncated": False,
                "error": f"SQL error: {e}"}


def run_query_markdown(sql: str, data_dir, max_rows: int = 100) -> str:
    """Same guarded query, formatted as a Markdown table (for the agent tool)."""
    r = run_query(sql, data_dir, max_rows)
    if r["error"]:
        return r["error"]
    if not r["rows"]:
        return "Query ran successfully but returned no rows."
    cols = r["columns"]
    out = "| " + " | ".join(str(c) for c in cols) + " |\n"
    out += "|" + "|".join("---" for _ in cols) + "|\n"
    for row in r["rows"]:
        out += "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n"
    note = f"\n\n_Showing {len(r['rows'])} of {r['total']} rows._" if r["truncated"] else ""
    return f"Query returned **{r['total']}** rows.\n\n{out}{note}"
