"""
Pet Store Supply Chain Intelligence MCP Server
Transport: SSE over HTTP (FastAPI + uvicorn)

Exposes 50 MCP tools:

  DATABASE TOOLS (original):
   1. get_inventory_status      — current inventory levels and risk classification
   2. get_demand_forecast       — TFT/CatBoost P10/P50/P90 demand forecast
   3. query_mysql               — run arbitrary read-only SQL on MySQL
   4. query_postgres            — run arbitrary read-only SQL on PostgreSQL
   5. get_supplier_info         — supplier performance data
   6. get_knowledge_base        — supply chain policies and guidelines
   7. log_forecast_to_postgres  — write a forecast result to PostgreSQL
   8. create_inventory_alert    — write an alert to PostgreSQL
   9. get_active_alerts         — fetch unresolved alerts from PostgreSQL
  10. get_monthly_kpis          — fetch aggregated KPIs from PostgreSQL
  11. get_stockout_risk         — SKUs that will stock out within N days
  12. get_reorder_list          — all SKUs needing reorder today
  13. get_demand_trends         — demand trend analysis across all SKUs
  14. get_regional_inventory    — inventory by region and category
  15. get_supply_chain_dashboard— company-wide health overview
  16. get_sku_360               — complete 360° profile of one SKU
  17. get_supplier_ranking      — suppliers ranked by reliability
  18. compare_categories        — side-by-side category comparison
  19. test_mysql_connection     — verify MySQL credentials
  20. test_postgres_connection  — verify PostgreSQL credentials

  ANALYSIS & CODE TOOLS:
  21. web_search               — Google search (SerpAPI + DuckDuckGo fallback)
  22. python_repl              — execute Python/pandas in a sandboxed REPL
  23. data_quality             — full data audit: nulls, negatives, outliers, anomalies

  INTELLIGENCE TOOLS:
  24. get_brand_performance            — brand revenue, margin, return rate, stockout days
  25. get_franchise_inventory_comparison — store/region inventory health comparison
  26. get_seasonal_demand_calendar     — Indian festival demand calendar with pre-stock recs
  27. get_cold_chain_monitor           — temperature breaches, expiry risk, waste value
  28. get_supplier_lead_time_tracker   — actual vs promised LT, OTD trend, underperformers
  29. get_return_rate_analysis         — return rates by category/brand, top reasons
  30. get_dead_stock_analysis          — dead stock identification, locked value, clearance pricing
  31. get_competitive_price_analysis   — Pet Store vs competitor price gap analysis
  32. get_new_product_launch_readiness — launch health score, demand ramp, early stockouts
  33. get_customer_segmentation_insights — segment LTV, AOV, frequency, channel preference
  34. generate_purchase_order          — full Pet Store purchase order with supplier grouping
  35. get_promotion_inventory_impact   — demand lift, stockouts, restock lag per promo
  36. get_channel_revenue_attribution  — Online/Offline/App revenue, margin, top SKUs
  37. get_markdown_optimization        — overstock clearance discount recommendations
  38. get_marketing_campaign_recommendations — top 5 categories to promote / avoid
  39. get_inventory_financial_summary  — CFO-level inventory value, margin, working capital
  40. get_customer_cohort_demand_analysis — quarterly cohort LTV, retention, top products
  41. get_store_level_demand_intelligence — 67-store demand intelligence, rebalancing
  42. get_supplier_negotiation_brief   — leverage score, YoY volume, negotiation talking points
  43. get_product_recommendation       — pet-specific product recommendations (breed/age/health)
  44. get_store_inventory_breakdown    — per-store live inventory from DB (city/region/risk)

Usage:
  uvicorn mcp_server.server:app --host 0.0.0.0 --port 8000

Environment variables (set in .env):
  MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
  PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB
  MCP_AUTH_TOKEN   (optional bearer token for auth)
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import inspect
import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

# Optional DB drivers graceful degradation if not installed / no creds
try:
    import aiomysql

    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import asyncpg

    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

# Paths
BASE_DIR = Path(__file__).parent.parent
HUFT_DATA_DIR = BASE_DIR / "data"
CSV_PATH = HUFT_DATA_DIR / "huft_daily_demand.csv"
PRODUCTS_CSV = HUFT_DATA_DIR / "huft_products.csv"
STORES_CSV = HUFT_DATA_DIR / "huft_stores.csv"
CUSTOMERS_CSV = HUFT_DATA_DIR / "huft_customers.csv"
PROMOTIONS_CSV = HUFT_DATA_DIR / "huft_promotions.csv"
TRANSACTIONS_CSV = HUFT_DATA_DIR / "huft_sales_transactions.csv"
RETURNS_CSV = HUFT_DATA_DIR / "huft_returns.csv"
SUPPLIER_PERF_CSV = HUFT_DATA_DIR / "huft_supplier_performance.csv"
COLD_CHAIN_CSV = HUFT_DATA_DIR / "huft_cold_chain.csv"

# DB config from env
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")

# BUG-048 fix: use ContextVar so each request/thread gets its own credential copy,
# eliminating the race condition where User A's set_session_creds() overwrites User B's.
_ctx_mysql_creds: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "_ctx_mysql_creds", default={}
)
_ctx_pg_creds: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "_ctx_pg_creds", default={}
)
# Fallback globals for the single-user / direct-call case
_session_mysql_creds: dict = {}
_session_pg_creds: dict = {}


def set_session_creds(mysql_creds: dict, pg_creds: dict) -> None:
    """Called by the Gradio UI before each agent run to push DB credentials.
    Stores in both a ContextVar (per-request isolation) and the global fallback."""
    global _session_mysql_creds, _session_pg_creds
    _session_mysql_creds = mysql_creds or {}
    _session_pg_creds = pg_creds or {}
    _ctx_mysql_creds.set(mysql_creds or {})
    _ctx_pg_creds.set(pg_creds or {})


def get_session_mysql_creds() -> dict:
    # ContextVar takes precedence; fall back to global if context not set
    ctx = _ctx_mysql_creds.get({})
    return ctx if ctx else _session_mysql_creds


def get_session_pg_creds() -> dict:
    ctx = _ctx_pg_creds.get({})
    return ctx if ctx else _session_pg_creds


# ── Tool Registry ─────────────────────────────────────────────────────────────
# Maps tool name → {schema, handler, cacheable, cache_ttl_seconds}
# Populated by the @tool decorator below.  MCP_TOOLS is derived from this.
_TOOL_REGISTRY: dict[str, dict] = {}

# Tool result cache: (tool_name, args_hash) → (result_str, expiry_timestamp)
# Only populated for tools registered with cacheable=True.
_TOOL_CACHE: dict[tuple, tuple[str, float]] = {}
_TOOL_CACHE_DEFAULT_TTL = 60  # seconds — override per-tool via cache_ttl


def tool(
    name: str,
    description: str,
    input_schema: dict,
    cacheable: bool = False,
    cache_ttl: int = _TOOL_CACHE_DEFAULT_TTL,
) -> Callable:
    """
    Decorator that registers a tool function in _TOOL_REGISTRY.

    Parameters
    ----------
    name          : MCP tool name (must be unique)
    description   : shown to the LLM — be precise and actionable
    input_schema  : JSON Schema dict for the tool's parameters
    cacheable     : if True, identical (name, args) calls are served from cache
                    for `cache_ttl` seconds — suitable for pure read-only CSV tools
    cache_ttl     : how long (seconds) to keep a cached result (default 60 s)
    """

    def decorator(fn: Callable) -> Callable:
        _TOOL_REGISTRY[name] = {
            "schema": {
                "name": name,
                "description": description,
                "inputSchema": input_schema,
            },
            "handler": fn,
            "is_async": inspect.iscoroutinefunction(fn),
            "cacheable": cacheable,
            "cache_ttl": cache_ttl,
        }
        return fn

    return decorator


def _args_cache_key(tool_name: str, args: dict) -> tuple:
    """Stable hash key for (tool_name, args) — safe across Python restarts."""
    args_json = json.dumps(args, sort_keys=True, default=str)
    args_hash = hashlib.md5(args_json.encode()).hexdigest()
    return (tool_name, args_hash)


async def _call_registered_tool(name: str, args: dict) -> str:
    """
    Dispatch a call to a registered tool, respecting the TTL cache.
    Falls back gracefully to a TOOL_ERROR string on exception.
    """
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        known = sorted(_TOOL_REGISTRY.keys())
        return f"TOOL_ERROR: Unknown tool '{name}'. Available: {known}"

    # Check cache
    if entry["cacheable"]:
        key = _args_cache_key(name, args)
        cached = _TOOL_CACHE.get(key)
        if cached is not None:
            result, expiry = cached
            if time.monotonic() < expiry:
                return result
        # Cache miss — fall through and populate after call

    try:
        fn = entry["handler"]
        if entry["is_async"]:
            result = await fn(**args)
        else:
            result = fn(**args)
    except TypeError as exc:
        # Bad args — return structured error so the LLM can self-correct
        sig = inspect.signature(entry["handler"])
        return (
            f"TOOL_ERROR [{name}]: Invalid arguments — {exc}.\n"
            f"Expected signature: {name}{sig}\n"
            f"Received args: {list(args.keys())}"
        )
    except Exception as exc:
        tb_short = traceback.format_exc(limit=3)
        return (
            f"TOOL_ERROR [{name}]: {type(exc).__name__}: {exc}\n\n"
            f"What this means:\n"
            f"  - The tool '{name}' failed to execute.\n"
            f"  - Possible causes: database not connected, bad SQL syntax, "
            f"missing data, or invalid arguments.\n\n"
            f"Suggested alternatives:\n"
            f"  - If this was a SQL query: try python_repl with the same logic on `df`\n"
            f"  - If this was a DB tool: check database connection with "
            f"test_mysql_connection or test_postgres_connection\n"
            f"  - If data is unavailable: answer from CSV cache using python_repl or "
            f"built-in inventory tools\n\n"
            f"Technical detail (for debugging):\n{tb_short}"
        )

    # Populate cache if applicable
    if entry["cacheable"]:
        key = _args_cache_key(name, args)
        _TOOL_CACHE[key] = (result, time.monotonic() + entry["cache_ttl"])

    return result


def invalidate_tool_cache(tool_name: str | None = None) -> None:
    """
    Clear the tool result cache.
    Pass a tool_name to clear only that tool's entries, or None to clear all.
    """
    global _TOOL_CACHE
    if tool_name is None:
        _TOOL_CACHE.clear()
    else:
        _TOOL_CACHE = {k: v for k, v in _TOOL_CACHE.items() if k[0] != tool_name}


# MCP_TOOLS is now auto-derived from the registry — no manual list needed.
# The property is re-evaluated lazily after all @tool decorators have run.
def _get_mcp_tools() -> list[dict]:
    return [entry["schema"] for entry in _TOOL_REGISTRY.values()]


# In-memory CSV cache with mtime-based invalidation
_df_cache: pd.DataFrame | None = None
_df_cache_mtime: float = 0.0
import threading as _threading

_df_cache_lock = _threading.Lock()


def _csv_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def get_df() -> pd.DataFrame:
    global _df_cache, _df_cache_mtime
    current_mtime = _csv_mtime(CSV_PATH)
    if _df_cache is not None and current_mtime == _df_cache_mtime:
        return _df_cache  # fast path — no lock needed
    with _df_cache_lock:  # slow path — double-checked locking
        current_mtime = _csv_mtime(CSV_PATH)  # re-read inside lock
        if _df_cache is None or current_mtime != _df_cache_mtime:
            if CSV_PATH.exists():
                _df_cache = pd.read_csv(CSV_PATH, parse_dates=["date"])
                _df_cache = _normalise_demand_df(_df_cache)
                _df_cache_mtime = current_mtime
            else:
                raise FileNotFoundError(
                    f"CSV not found at {CSV_PATH}. Run: python -m data.generate_data"
                )
    return _df_cache


def _normalise_demand_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expected columns exist regardless of data source (CSV or MySQL).
    Columns present in huft_daily_demand.csv but absent from the MySQL daily_demand
    table are backfilled with sensible defaults so tools never crash with KeyError.
    """
    # price alias
    if "price_inr" in df.columns and "price_usd" not in df.columns:
        df["price_usd"] = df["price_inr"]
    elif "price_usd" in df.columns and "price_inr" not in df.columns:
        df["price_inr"] = df["price_usd"]

    # financial columns absent from MySQL schema — fill with 0 so arithmetic
    # still runs; tools that need real values will show 0 rather than crashing
    for col, default in [
        ("cost_inr", 0.0),
        ("margin_pct", 0.0),
        ("price_inr", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # region normalisation
    if "region" not in df.columns:
        df["region"] = "India"

    return df


# ── Lazy-loading helpers for supplementary Pet Store CSVs ────────────────────

_products_cache: pd.DataFrame | None = None
_stores_cache: pd.DataFrame | None = None
_customers_cache: pd.DataFrame | None = None
_promotions_cache: pd.DataFrame | None = None
_transactions_cache: pd.DataFrame | None = None
_returns_cache: pd.DataFrame | None = None
_supplier_perf_cache: pd.DataFrame | None = None
_cold_chain_cache: pd.DataFrame | None = None


def get_products() -> pd.DataFrame:
    global _products_cache
    if _products_cache is None:
        if PRODUCTS_CSV.exists():
            _products_cache = pd.read_csv(PRODUCTS_CSV)
        else:
            raise FileNotFoundError(f"Products CSV not found: {PRODUCTS_CSV}")
    return _products_cache


def get_stores() -> pd.DataFrame:
    global _stores_cache
    if _stores_cache is None:
        if STORES_CSV.exists():
            _stores_cache = pd.read_csv(STORES_CSV)
        else:
            raise FileNotFoundError(f"Stores CSV not found: {STORES_CSV}")
    return _stores_cache


def get_customers() -> pd.DataFrame:
    global _customers_cache
    if _customers_cache is None:
        if CUSTOMERS_CSV.exists():
            _customers_cache = pd.read_csv(CUSTOMERS_CSV, parse_dates=["joined_date"])
        else:
            raise FileNotFoundError(f"Customers CSV not found: {CUSTOMERS_CSV}")
    return _customers_cache


def get_promotions() -> pd.DataFrame:
    global _promotions_cache
    if _promotions_cache is None:
        if PROMOTIONS_CSV.exists():
            _promotions_cache = pd.read_csv(
                PROMOTIONS_CSV, parse_dates=["start_date", "end_date"]
            )
        else:
            raise FileNotFoundError(f"Promotions CSV not found: {PROMOTIONS_CSV}")
    return _promotions_cache


def get_transactions() -> pd.DataFrame:
    global _transactions_cache
    if _transactions_cache is None:
        if TRANSACTIONS_CSV.exists():
            _transactions_cache = pd.read_csv(TRANSACTIONS_CSV, parse_dates=["date"])
        else:
            raise FileNotFoundError(f"Transactions CSV not found: {TRANSACTIONS_CSV}")
    return _transactions_cache


def get_returns() -> pd.DataFrame:
    global _returns_cache
    if _returns_cache is None:
        if RETURNS_CSV.exists():
            _returns_cache = pd.read_csv(RETURNS_CSV, parse_dates=["return_date"])
        else:
            raise FileNotFoundError(f"Returns CSV not found: {RETURNS_CSV}")
    return _returns_cache


def get_supplier_perf() -> pd.DataFrame:
    global _supplier_perf_cache
    if _supplier_perf_cache is None:
        if SUPPLIER_PERF_CSV.exists():
            _supplier_perf_cache = pd.read_csv(
                SUPPLIER_PERF_CSV, parse_dates=["review_month"]
            )
        else:
            raise FileNotFoundError(
                f"Supplier performance CSV not found: {SUPPLIER_PERF_CSV}"
            )
    return _supplier_perf_cache


def get_cold_chain() -> pd.DataFrame:
    global _cold_chain_cache
    if _cold_chain_cache is None:
        if COLD_CHAIN_CSV.exists():
            _cold_chain_cache = pd.read_csv(
                COLD_CHAIN_CSV, parse_dates=["date", "expiry_date"]
            )
        else:
            raise FileNotFoundError(f"Cold chain CSV not found: {COLD_CHAIN_CSV}")
    return _cold_chain_cache


# DB credential helpers


def _resolve_mysql_cfg(creds: dict | None) -> dict:
    """
    Build a MySQL connection config dict.
    Priority: per-call creds dict > environment variables > hard-coded defaults.
    creds keys: host, port, user, password, db
    """
    c = creds or {}
    return {
        "host": c.get("host") or os.getenv("MYSQL_HOST", "localhost"),
        "port": int(c.get("port") or os.getenv("MYSQL_PORT", 3306)),
        "user": c.get("user") or os.getenv("MYSQL_USER", "root"),
        "password": c.get("password") or os.getenv("MYSQL_PASSWORD", ""),
        "db": c.get("db") or os.getenv("MYSQL_DB", "pet_store_scm"),
        "autocommit": True,
    }


def _resolve_pg_dsn(creds: dict | None) -> str:
    """
    Build a PostgreSQL DSN string.
    Priority: per-call creds dict > environment variables > hard-coded defaults.
    creds keys: host, port, user, password, db
    """
    c = creds or {}
    user = c.get("user") or os.getenv("PG_USER", "postgres")
    password = c.get("password") or os.getenv("PG_PASSWORD", "")
    host = c.get("host") or os.getenv("PG_HOST", "localhost")
    port = c.get("port") or os.getenv("PG_PORT", "5432")
    db = c.get("db") or os.getenv("PG_DB", "pet_store_scm")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _has_mysql_creds(creds: dict | None) -> bool:
    """Return True if MySQL credentials are available (from call or env)."""
    c = creds or {}
    return bool(
        c.get("host")
        or c.get("password")
        or os.getenv("MYSQL_HOST")
        or os.getenv("MYSQL_PASSWORD")
    )


def _has_pg_creds(creds: dict | None) -> bool:
    """Return True if PostgreSQL credentials are available (from call or env)."""
    c = creds or {}
    return bool(
        c.get("host")
        or c.get("password")
        or os.getenv("PG_HOST")
        or os.getenv("PG_PASSWORD")
    )


# DB connection helpers all accept optional per-call credentials


async def mysql_query(
    sql: str, args: tuple = (), creds: dict | None = None
) -> list[dict]:
    if not MYSQL_AVAILABLE:
        return [{"error": "aiomysql not installed. Run: pip install aiomysql"}]
    if not _has_mysql_creds(creds):
        return [
            {
                "error": (
                    "MySQL credentials not configured. "
                    "Fill in the MySQL settings panel in the UI, "
                    "or set MYSQL_HOST and MYSQL_PASSWORD in the .env file."
                )
            }
        ]
    conn = None
    try:
        cfg = _resolve_mysql_cfg(creds)
        conn = await aiomysql.connect(**cfg)
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        return [{"error": str(exc)}]
    finally:
        if conn is not None:
            conn.close()


async def pg_query(sql: str, args: tuple = (), creds: dict | None = None) -> list[dict]:
    if not PG_AVAILABLE:
        return [{"error": "asyncpg not installed. Run: pip install asyncpg"}]
    if not _has_pg_creds(creds):
        return [
            {
                "error": (
                    "PostgreSQL credentials not configured. "
                    "Fill in the PostgreSQL settings panel in the UI, "
                    "or set PG_HOST and PG_PASSWORD in the .env file."
                )
            }
        ]
    conn = None
    try:
        dsn = _resolve_pg_dsn(creds)
        conn = await asyncpg.connect(dsn)
        rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]
    except Exception as exc:
        return [{"error": str(exc)}]
    finally:
        if conn is not None:
            await conn.close()


async def pg_execute(sql: str, args: tuple = (), creds: dict | None = None) -> str:
    if not PG_AVAILABLE:
        return "asyncpg not installed. Run: pip install asyncpg"
    if not _has_pg_creds(creds):
        return (
            "ERROR: PostgreSQL credentials not configured. "
            "Fill in the PostgreSQL settings panel in the UI, "
            "or set PG_HOST and PG_PASSWORD in the .env file."
        )
    # BUG-4 fix: close connection in finally so it always releases even on SQL errors
    conn = None
    try:
        dsn = _resolve_pg_dsn(creds)
        conn = await asyncpg.connect(dsn)
        result = await conn.execute(sql, *args)
        return result
    except Exception as exc:
        return f"ERROR: {exc}"
    finally:
        if conn is not None:
            await conn.close()


# Connection test helpers (used by the UI "Test Connection" buttons)


async def test_mysql_connection(creds: dict | None = None) -> dict:
    """Test MySQL connectivity. Returns {ok: bool, message: str, details: dict}."""
    if not MYSQL_AVAILABLE:
        return {
            "ok": False,
            "message": "aiomysql not installed. Run: pip install aiomysql",
            "details": {},
        }
    if not _has_mysql_creds(creds):
        return {"ok": False, "message": "No MySQL credentials provided.", "details": {}}
    cfg = _resolve_mysql_cfg(creds)
    conn = None
    try:
        conn = await aiomysql.connect(**cfg)
        async with conn.cursor() as cur:
            await cur.execute("SELECT VERSION() AS version, DATABASE() AS db_name")
            row = await cur.fetchone()
        return {
            "ok": True,
            "message": f"Connected to MySQL at {cfg['host']}:{cfg['port']}",
            "details": {
                "host": cfg["host"],
                "port": cfg["port"],
                "user": cfg["user"],
                "database": cfg["db"],
                "version": row[0] if row else "unknown",
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": str(exc),
            "details": {"host": cfg.get("host"), "port": cfg.get("port")},
        }
    finally:
        if conn is not None:
            conn.close()


async def test_postgres_connection(creds: dict | None = None) -> dict:
    """Test PostgreSQL connectivity. Returns {ok: bool, message: str, details: dict}."""
    if not PG_AVAILABLE:
        return {
            "ok": False,
            "message": "asyncpg not installed. Run: pip install asyncpg",
            "details": {},
        }
    if not _has_pg_creds(creds):
        return {
            "ok": False,
            "message": "No PostgreSQL credentials provided.",
            "details": {},
        }
    dsn = _resolve_pg_dsn(creds)
    c = creds or {}
    host = c.get("host") or os.getenv("PG_HOST", "localhost")
    port = c.get("port") or os.getenv("PG_PORT", "5432")
    conn = None
    try:
        conn = await asyncpg.connect(dsn)
        row = await conn.fetchrow("SELECT version(), current_database()")
        return {
            "ok": True,
            "message": f"Connected to PostgreSQL at {host}:{port}",
            "details": {
                "host": host,
                "port": port,
                "database": row["current_database"] if row else "unknown",
                "version": (row["version"] if row else "unknown")[:60],
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": str(exc),
            "details": {"host": host, "port": port},
        }
    finally:
        if conn is not None:
            await conn.close()


# TOOL IMPLEMENTATIONS


def tool_get_inventory_status(sku_id: str | None = None, top_n: int = 10) -> str:
    """Returns current inventory status from the CSV data."""
    df = get_df()
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()

    # avg demand over last 30 days
    cutoff = latest_date - pd.Timedelta(days=30)
    recent = df[df["date"] >= cutoff].groupby("sku_id")["demand"].mean().reset_index()
    recent.columns = ["sku_id", "avg_daily_demand"]

    merged = latest.merge(recent, on="sku_id")
    merged["days_of_supply"] = (
        (merged["inventory"] / merged["avg_daily_demand"].replace(0, np.nan))
        .fillna(0)
        .round(1)
    )

    def risk(row):
        dos = row["days_of_supply"]
        lt = row["lead_time_days"]
        if dos < lt:
            return "CRITICAL"
        if dos < 2 * lt:
            return "WARNING"
        return "OK"

    merged["risk"] = merged.apply(risk, axis=1)

    if sku_id:
        sku_id = sku_id.upper()
        row = merged[merged["sku_id"] == sku_id]
        if row.empty:
            return f"SKU '{sku_id}' not found. Valid prefixes: DOG, CAT, MED, ACC."
        r = row.iloc[0]
        return (
            f"=== Inventory Status: {r['sku_id']} — {r['name']} ===\n"
            f"Category:        {r['category']} / {r['subcategory']}\n"
            f"Supplier:        {r['supplier']} ({r['region']})\n"
            f"Current Inventory: {int(r['inventory']):,} units\n"
            f"Avg Daily Demand:  {r['avg_daily_demand']:.1f} units/day\n"
            f"Days of Supply:    {r['days_of_supply']:.1f} days\n"
            f"Lead Time:         {int(r['lead_time_days'])} days\n"
            f"Risk Status:       {r['risk']}\n"
            f"Price:             ₹{float(r.get('price_inr', r.get('price_usd', 0))):,.0f}\n"
            f"Data as of:        {latest_date.date()}"
        )

    # top-N at risk
    at_risk = merged[merged["risk"] != "OK"].sort_values("days_of_supply").head(top_n)
    if at_risk.empty:
        return f"All {len(merged)} SKUs have adequate inventory (as of {latest_date.date()})."

    lines = [f"=== Top {len(at_risk)} At-Risk SKUs (as of {latest_date.date()}) ===\n"]
    for _, r in at_risk.iterrows():
        lines.append(
            f"[{r['risk']}] {r['sku_id']} – {r['name']}\n"
            f"  Inventory: {int(r['inventory']):,} units | "
            f"Avg Demand: {r['avg_daily_demand']:.1f}/day | "
            f"Days of Supply: {r['days_of_supply']:.1f} | "
            f"Lead Time: {int(r['lead_time_days'])}d\n"
        )
    return "\n".join(lines)


def _build_risk_df() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Shared helper: latest inventory merged with 30-day avg demand + risk labels.
    Returns (risk_dataframe, latest_date)."""
    df = get_df()
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    cutoff = latest_date - pd.Timedelta(days=30)
    avg_demand = (
        df[df["date"] >= cutoff]
        .groupby("sku_id")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "avg_daily_demand"})
    )
    m = latest.merge(avg_demand, on="sku_id")
    m["days_of_supply"] = (
        (m["inventory"] / m["avg_daily_demand"].replace(0, np.nan)).fillna(0).round(1)
    )
    m["risk"] = m.apply(
        lambda r: (
            "CRITICAL"
            if r["days_of_supply"] < r["lead_time_days"]
            else "WARNING"
            if r["days_of_supply"] < 2 * r["lead_time_days"]
            else "OK"
        ),
        axis=1,
    )
    return m, latest_date


def tool_get_regional_inventory(category: str | None = None) -> str:
    """
    Returns a full inventory breakdown grouped by region, with per-region risk
    counts (CRITICAL / WARNING / OK) and at-risk SKU details.
    Optionally filter to one product category.
    Use this for ANY question about regional inventory, stock levels by location,
    or 'which regions are most at risk'.
    """
    m, latest_date = _build_risk_df()
    if category:
        m = m[m["category"].str.contains(category, case=False, na=False)]
        if m.empty:
            return f"No SKUs found for category matching '{category}'."
    scope = category or "All Categories"

    # Regional summary
    agg = (
        m.groupby("region")
        .agg(
            total_skus=("sku_id", "count"),
            critical=("risk", lambda x: (x == "CRITICAL").sum()),
            warning=("risk", lambda x: (x == "WARNING").sum()),
            ok=("risk", lambda x: (x == "OK").sum()),
            avg_dos=("days_of_supply", "mean"),
            total_inventory=("inventory", "sum"),
        )
        .reset_index()
    )

    lines = [f"=== Inventory by Region — {scope} (as of {latest_date.date()}) ===\n"]
    for _, r in agg.sort_values("critical", ascending=False).iterrows():
        health = (
            "NEEDS ATTENTION"
            if r["critical"] > 0
            else "MONITOR"
            if r["warning"] > 0
            else "HEALTHY"
        )
        lines.append(
            f"[{health}] {r['region']}\n"
            f"  SKUs: {int(r['total_skus'])} | "
            f"CRITICAL: {int(r['critical'])} | WARNING: {int(r['warning'])} | OK: {int(r['ok'])}\n"
            f"  Total Inventory: {int(r['total_inventory']):,} units | "
            f"Avg Days of Supply: {r['avg_dos']:.1f} days\n"
        )

    at_risk = m[m["risk"] != "OK"].sort_values(["region", "days_of_supply"])
    if not at_risk.empty:
        lines.append("\n── At-Risk SKUs by Region ──")
        for region, grp in at_risk.groupby("region"):
            lines.append(f"\n  {region}:")
            for _, s in grp.iterrows():
                lines.append(
                    f"    [{s['risk']}] {s['sku_id']} – {s['name']} | "
                    f"Inventory: {int(s['inventory']):,} | "
                    f"DoS: {s['days_of_supply']:.1f}d | Lead Time: {int(s['lead_time_days'])}d"
                )
    else:
        lines.append(f"\n✓ No at-risk SKUs in {scope}.")
    return "\n".join(str(l) for l in lines)


def tool_get_supply_chain_dashboard() -> str:
    """
    Returns a complete supply chain snapshot in ONE call: total SKU counts,
    risk breakdown per category, top critical SKUs, and key recommendations.
    Use this for broad questions like 'what is the overall status', 'what needs
    attention today', 'give me a full report', or 'executive summary'.
    """
    m, latest_date = _build_risk_df()
    total = len(m)
    critical = int((m["risk"] == "CRITICAL").sum())
    warning = int((m["risk"] == "WARNING").sum())
    ok = int((m["risk"] == "OK").sum())
    avg_dos = float(m["days_of_supply"].mean())

    # Per-category breakdown
    cat_agg = (
        m.groupby("category")
        .agg(
            skus=("sku_id", "count"),
            critical=("risk", lambda x: (x == "CRITICAL").sum()),
            warning=("risk", lambda x: (x == "WARNING").sum()),
            avg_dos=("days_of_supply", "mean"),
        )
        .reset_index()
        .sort_values("critical", ascending=False)
    )

    # Top critical SKUs
    top_crit = m[m["risk"] == "CRITICAL"].sort_values("days_of_supply").head(5)

    # BUG-030 fix: ensure percentages sum to 100% by using total as denominator
    _total_safe = max(total, 1)
    lines = [
        f"╔══ SUPPLY CHAIN DASHBOARD (as of {latest_date.date()}) ══╗\n",
        f"  Total SKUs  : {total}",
        f"  CRITICAL    : {critical}  ({critical / _total_safe * 100:.0f}%)",
        f"  WARNING     : {warning}   ({warning / _total_safe * 100:.0f}%)",
        f"  OK          : {ok}   ({ok / _total_safe * 100:.0f}%)",
        f"  Avg DoS     : {avg_dos:.1f} days\n",
        "── By Category ──",
    ]
    for _, r in cat_agg.iterrows():
        flag = "⚠" if r["critical"] > 0 else ("!" if r["warning"] > 0 else "✓")
        lines.append(
            f"  {flag} {r['category']}: {int(r['skus'])} SKUs | "
            f"Critical: {int(r['critical'])} | Warning: {int(r['warning'])} | "
            f"Avg DoS: {r['avg_dos']:.1f}d"
        )

    if not top_crit.empty:
        lines.append("\n── Top Critical SKUs (immediate action needed) ──")
        for _, s in top_crit.iterrows():
            reorder_qty = int(s["avg_daily_demand"] * s["lead_time_days"] * 1.5)
            lines.append(
                f"  {s['sku_id']} – {s['name']} ({s['category']})\n"
                f"    Inventory: {int(s['inventory']):,} | DoS: {s['days_of_supply']:.1f}d | "
                f"Supplier: {s['supplier']} | Suggested reorder: {reorder_qty:,} units"
            )
    else:
        lines.append("\n✓ No critical SKUs — supply chain is healthy.")
    return "\n".join(str(l) for l in lines)


def tool_get_sku_360(sku_id: str) -> str:
    """
    Returns a complete 360° profile of a single SKU: current inventory, risk status,
    30-day demand history stats, supplier info, lead time, price, and a reorder
    recommendation with suggested quantity.
    Use this whenever you need EVERYTHING about one specific SKU in one call.
    """
    df = get_df()
    sku_id = sku_id.upper().strip()
    sku_df = df[df["sku_id"] == sku_id].sort_values("date")
    if sku_df.empty:
        return f"SKU '{sku_id}' not found. Valid prefixes: DOG, CAT, MED, ACC."

    latest = sku_df.iloc[-1]
    hist_30 = sku_df.tail(30)["demand"]
    hist_90 = sku_df.tail(90)["demand"]

    avg_30 = float(hist_30.mean())
    avg_90 = float(hist_90.mean())
    std_30 = float(hist_30.std())
    # Use 90-day std for safety stock — consistent with build_forecast_fig
    std_90 = float(hist_90.std())
    inv = int(latest["inventory"])
    lt = int(latest["lead_time_days"])
    dos = round(inv / avg_30, 1) if avg_30 > 0 else 0
    risk = "CRITICAL" if dos < lt else "WARNING" if dos < 2 * lt else "OK"

    safety_stock = round(1.65 * std_90 * (lt**0.5))
    reorder_pt = round(avg_90 * lt + safety_stock)
    usable_inv = max(0, inv - safety_stock)
    reorder_qty = max(0, round(avg_90 * lt) - usable_inv)
    trend = (
        "↑ increasing"
        if avg_30 > avg_90 * 1.05
        else "↓ decreasing"
        if avg_30 < avg_90 * 0.95
        else "→ stable"
    )

    lines = [
        f"╔══ SKU 360°: {sku_id} — {latest['name']} ══╗\n",
        f"  Category   : {latest['category']} / {latest['subcategory']}",
        f"  Supplier   : {latest['supplier']}  |  Region: {latest['region']}",
        f"  Price (₹)  : ₹{float(latest.get('price_inr', latest.get('price_usd', 0))):,.0f}\n",
        f"── Current Inventory ──",
        f"  Stock      : {inv:,} units",
        f"  Days of Supply: {dos} days  |  Lead Time: {lt} days",
        f"  Risk Status: {risk}\n",
        f"── Demand Profile ──",
        f"  30-day avg : {avg_30:.1f} units/day  (±{std_30:.1f})",
        f"  90-day avg : {avg_90:.1f} units/day",
        f"  Trend      : {trend}\n",
        f"── Reorder Analysis ──",
        f"  Safety Stock : {safety_stock:,} units  (Z=1.65, σ={std_90:.1f} [90d], LT={lt}d)",
        f"  Reorder Point: {reorder_pt:,} units",
        f"  Suggested Qty: {reorder_qty:,} units",
        f"\n  Recommendation: {'⚠ REORDER NOW' if inv < reorder_pt else '✓ Monitor — reorder when below ' + str(reorder_pt) + ' units'}",
    ]
    return "\n".join(str(l) for l in lines)


def tool_get_supplier_ranking() -> str:
    """
    Returns all suppliers ranked by reliability: lead time, number of SKUs supplied,
    and at-risk SKU counts. Use this for questions about supplier performance,
    'which supplier is best/worst', or 'who should I reorder from'.
    """
    m, latest_date = _build_risk_df()
    sup = (
        m.groupby("supplier")
        .agg(
            skus=("sku_id", "count"),
            critical=("risk", lambda x: (x == "CRITICAL").sum()),
            warning=("risk", lambda x: (x == "WARNING").sum()),
            avg_lt=("lead_time_days", "mean"),
            avg_dos=("days_of_supply", "mean"),
            categories=("category", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values(["critical", "avg_lt"])
    )

    lines = [f"=== Supplier Ranking (as of {latest_date.date()}) ===\n"]
    for rank, (_, r) in enumerate(sup.iterrows(), 1):
        grade = (
            "A"
            if r["critical"] == 0 and r["avg_lt"] <= 10
            else "B"
            if r["critical"] == 0
            else "C"
        )
        lines.append(
            f"#{rank} [{grade}] {r['supplier']}\n"
            f"   SKUs supplied : {int(r['skus'])}  ({r['categories']})\n"
            f"   Avg Lead Time : {r['avg_lt']:.1f} days\n"
            f"   At-Risk SKUs  : CRITICAL {int(r['critical'])} | WARNING {int(r['warning'])}\n"
            f"   Avg Days of Supply: {r['avg_dos']:.1f} days\n"
        )
    return "\n".join(str(l) for l in lines)


def tool_compare_categories() -> str:
    """
    Returns a side-by-side comparison of all product categories: total SKUs,
    risk breakdown, average days of supply, and top at-risk items.
    Use this for questions like 'how are all categories doing', 'compare dog vs cat food',
    or 'which category needs the most attention'.
    """
    m, latest_date = _build_risk_df()
    agg = (
        m.groupby("category")
        .agg(
            skus=("sku_id", "count"),
            critical=("risk", lambda x: (x == "CRITICAL").sum()),
            warning=("risk", lambda x: (x == "WARNING").sum()),
            ok=("risk", lambda x: (x == "OK").sum()),
            avg_dos=("days_of_supply", "mean"),
            min_dos=("days_of_supply", "min"),
            total_inv=("inventory", "sum"),
        )
        .reset_index()
        .sort_values("critical", ascending=False)
    )

    lines = [f"=== Category Comparison (as of {latest_date.date()}) ===\n"]
    for _, r in agg.iterrows():
        health_score = round((r["ok"] / max(r["skus"], 1)) * 100)
        lines.append(
            f"── {r['category']} ──\n"
            f"   Health Score  : {health_score}%\n"
            f"   SKUs          : {int(r['skus'])} total | "
            f"CRITICAL: {int(r['critical'])} | WARNING: {int(r['warning'])} | OK: {int(r['ok'])}\n"
            f"   Avg Days Supply: {r['avg_dos']:.1f}d  |  Min: {r['min_dos']:.1f}d\n"
            f"   Total Inventory: {int(r['total_inv']):,} units\n"
        )
        worst = m[
            (m["category"] == r["category"]) & (m["risk"] == "CRITICAL")
        ].nsmallest(2, "days_of_supply")
        if not worst.empty:
            lines.append(
                f"   Urgent: "
                + " | ".join(
                    f"{s['sku_id']} ({s['days_of_supply']:.1f}d)"
                    for _, s in worst.iterrows()
                )
                + "\n"
            )
    return "\n".join(str(l) for l in lines)


def tool_get_stockout_risk(days: int = 14) -> str:
    """
    Returns every SKU that will run out of stock within the next N days,
    based on current inventory ÷ average daily demand (last 30 days).
    Ranked by urgency with supplier and suggested order quantity.
    """
    m, latest_date = _build_risk_df()
    m["days_until_stockout"] = (
        (m["inventory"] / m["avg_daily_demand"].replace(0, np.nan)).fillna(999).round(1)
    )

    at_risk = m[m["days_until_stockout"] <= days].sort_values("days_until_stockout")
    if at_risk.empty:
        return (
            f"✓ No stockouts expected in the next {days} days "
            f"(as of {latest_date.date()}). All inventory adequate."
        )

    lines = [
        f"=== Stockout Risk — Next {days} Days (as of {latest_date.date()}) ===\n",
        f"⚠ {len(at_risk)} SKUs at risk:\n",
    ]
    for _, r in at_risk.iterrows():
        urgency = (
            "CRITICAL" if r["days_until_stockout"] <= r["lead_time_days"] else "URGENT"
        )
        reorder_qty = int(r["avg_daily_demand"] * r["lead_time_days"] * 2)
        lines.append(
            f"[{urgency}] {r['sku_id']} – {r['name']}\n"
            f"  Stocks out in : {r['days_until_stockout']:.1f} days\n"
            f"  Lead Time     : {int(r['lead_time_days'])} days | "
            f"Inventory: {int(r['inventory']):,} units\n"
            f"  Supplier      : {r['supplier']}\n"
            f"  Suggested Order: {reorder_qty:,} units\n"
        )
    return "\n".join(str(l) for l in lines)


def tool_get_reorder_list() -> str:
    """
    Generates a complete, prioritised purchase order list for all SKUs at or
    below their reorder point.  For each SKU: supplier, urgency, suggested
    order quantity (covers lead time + safety stock), and estimated cost.
    This is the primary operational output for procurement teams.
    """
    m, latest_date = _build_risk_df()

    # Compute per-SKU std from the last 90 days of history — consistent with
    # build_forecast_fig and tool_get_sku_360. No CoV heuristic.
    df_full = get_df()
    std_map = df_full.groupby("sku_id")["demand"].apply(
        lambda x: float(x.iloc[-90:].std()) if len(x) >= 90 else float(x.std())
    )
    m["sigma"] = m["sku_id"].map(std_map).fillna(m["avg_daily_demand"] * 0.25)
    m["safety_stock"] = (1.65 * m["sigma"] * np.sqrt(m["lead_time_days"])).round(0)
    m["reorder_point"] = (
        m["avg_daily_demand"] * m["lead_time_days"] + m["safety_stock"]
    ).round(0)
    m["usable_inv"] = (m["inventory"] - m["safety_stock"]).clip(lower=0)
    m["order_qty"] = (
        (m["avg_daily_demand"] * m["lead_time_days"] - m["usable_inv"])
        .clip(lower=0)
        .round(0)
        .astype(int)
    )
    # BUG-009 fix: use cost_inr (landed cost) not retail price for PO cost estimate
    cost_col = (
        "cost_inr"
        if "cost_inr" in m.columns and m["cost_inr"].gt(0).any()
        else "price_usd"
    )
    m["est_cost"] = (m["order_qty"] * m[cost_col].fillna(0)).round(2)
    m["dos"] = (
        (m["inventory"] / m["avg_daily_demand"].replace(0, np.nan)).fillna(999).round(1)
    )

    needs = m[m["inventory"] <= m["reorder_point"]].sort_values("dos")
    if needs.empty:
        return f"✓ No reorders needed as of {latest_date.date()}. All SKUs above reorder point."

    total_cost = float(needs["est_cost"].sum())
    lines = [
        f"╔══ PURCHASE ORDER LIST — {latest_date.date()} ══╗\n",
        f"  {len(needs)} SKUs require reordering | Est. Total: ₹{total_cost:,.0f}\n",
        "── Ordered by Urgency ──\n",
    ]
    for i, (_, r) in enumerate(needs.iterrows(), 1):
        flag = "🔴 CRITICAL" if r["risk"] == "CRITICAL" else "🟡 WARNING"
        lines.append(
            f"{i:2}. {flag} | {r['sku_id']} – {r['name']}\n"
            f"     Supplier  : {r['supplier']}\n"
            f"     In Stock  : {int(r['inventory']):,} units | DoS: {r['dos']:.1f} days\n"
            f"     Order Qty : {r['order_qty']:,} units | Est. Cost: ₹{r['est_cost']:,.0f}\n"
        )
    return "\n".join(str(l) for l in lines)


def tool_get_demand_trends(days: int = 90) -> str:
    """
    Identifies SKUs with significantly increasing or decreasing demand by
    comparing average daily demand in the first vs second half of the period.
    Shows top growing and top declining SKUs with percentage change.
    Use for: 'which products are trending up/down', 'demand changes', YoY questions.
    """
    df = get_df()
    latest_date = df["date"].max()
    cutoff = latest_date - pd.Timedelta(days=days)
    period = df[df["date"] >= cutoff].copy()
    # BUG-024 fix: split exactly at the midpoint of the date range (not len//2)
    mid = cutoff + pd.Timedelta(days=days / 2)

    h1 = (
        period[period["date"] < mid]
        .groupby("sku_id")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "first_avg"})
    )
    h2 = (
        period[period["date"] >= mid]
        .groupby("sku_id")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "second_avg"})
    )

    trends = h1.merge(h2, on="sku_id")
    trends["change_pct"] = (
        (trends["second_avg"] - trends["first_avg"])
        / (trends["first_avg"] + 1e-6)
        * 100
    ).round(1)

    latest_skus = df[df["date"] == latest_date][
        ["sku_id", "name", "category"]
    ].drop_duplicates()
    trends = trends.merge(latest_skus, on="sku_id", how="left")

    growing = (
        trends[trends["change_pct"] > 10]
        .sort_values("change_pct", ascending=False)
        .head(10)
    )
    declining = trends[trends["change_pct"] < -10].sort_values("change_pct").head(10)
    stable = trends[(trends["change_pct"] >= -10) & (trends["change_pct"] <= 10)]

    lines = [
        f"=== Demand Trends — Last {days} Days (as of {latest_date.date()}) ===\n",
        f"  ↑ Growing : {len(growing)} SKUs | ↓ Declining: {len(declining)} SKUs "
        f"| → Stable: {len(stable)} SKUs\n",
    ]
    if not growing.empty:
        lines.append("── Top Growing SKUs ──")
        for _, r in growing.iterrows():
            lines.append(
                f"  ↑ {r['sku_id']} – {r['name']} ({r['category']})\n"
                f"    {r['first_avg']:.1f}/day → {r['second_avg']:.1f}/day  "
                f"(+{r['change_pct']:.1f}%)"
            )
    if not declining.empty:
        lines.append("\n── Top Declining SKUs ──")
        for _, r in declining.iterrows():
            lines.append(
                f"  ↓ {r['sku_id']} – {r['name']} ({r['category']})\n"
                f"    {r['first_avg']:.1f}/day → {r['second_avg']:.1f}/day  "
                f"({r['change_pct']:.1f}%)"
            )
    return "\n".join(str(l) for l in lines)


def tool_get_demand_forecast(sku_id: str, horizon_days: int = 30) -> str:
    """Returns a CatBoost demand forecast (P10/P50/P90) for a given SKU."""
    df = get_df()
    sku_id = sku_id.upper()
    sku_df = df[df["sku_id"] == sku_id].sort_values("date")

    if sku_df.empty:
        return f"SKU '{sku_id}' not found."

    name = sku_df["name"].iloc[0]
    if len(sku_df) < 30:
        return f"Not enough data to forecast for {sku_id} (need ≥ 30 days)."

    inv = int(sku_df["inventory"].iloc[-1])
    lead_time = int(sku_df["lead_time_days"].iloc[-1])

    # ── Try CatBoost model first ──────────────────────────────────────────
    method = "Statistical (mean ± 1.65σ)"
    try:
        from forecasting.ml_forecast import forecast as ml_forecast, is_trained

        if is_trained():
            preds = ml_forecast(sku_id, sku_df, horizon_days)
            p10 = preds["p10"]
            p50 = preds["p50"]
            p90 = preds["p90"]
            method = "CatBoost Quantile Regression"
        else:
            raise RuntimeError("Models not trained yet")
    except Exception:
        # ── Statistical fallback (BUG-036/037 fix) ───────────────────────
        # Bands widen with √t; trend from actual slope, not a fixed +5%.
        recent = sku_df["demand"].values[-60:]
        avg = float(np.mean(recent))
        std = float(np.std(recent)) if len(recent) > 1 else avg * 0.2
        if len(recent) >= 14:
            x = np.arange(len(recent), dtype=float)
            slope, _ = np.polyfit(x, recent, 1)
            trend_per_day = float(np.clip(slope / max(avg, 1e-6), -0.005, 0.005))
        else:
            trend_per_day = 0.0
        t = np.arange(1, horizon_days + 1, dtype=float)
        p50 = np.maximum(avg * (1.0 + trend_per_day * t), 0)
        expanding_std = std * np.sqrt(t)
        p10 = np.maximum(p50 - 1.65 * expanding_std, 0)
        p90 = p50 + 1.65 * expanding_std

    p10_total = float(p10.sum())
    p50_total = float(p50.sum())
    p90_total = float(p90.sum())
    p50_daily = float(p50.mean())

    # BUG-013 fix: use 90-day sigma consistently with tool_get_sku_360
    sigma = float(np.std(sku_df["demand"].values[-90:]))
    safety_stock = 1.65 * sigma * float(np.sqrt(lead_time))
    reorder_point = p50_daily * lead_time + safety_stock

    reorder_needed = inv < reorder_point
    rec = (
        f"REORDER RECOMMENDED: inventory ({inv:,}) < reorder point ({reorder_point:.0f}). "
        f"Order ~{p50_total:.0f} units (safety stock: {safety_stock:.0f} units)."
        if reorder_needed
        else f"Stock adequate. Reorder point: {reorder_point:.0f} units "
        f"(safety stock: {safety_stock:.0f} units)."
    )

    return (
        f"=== {horizon_days}-Day Demand Forecast: {sku_id} – {name} ===\n"
        f"Forecast Engine:   {method}\n\n"
        f"  Pessimistic (P10): {p10_total:,.0f} units total  ({p10.mean():.1f}/day)\n"
        f"  Expected    (P50): {p50_total:,.0f} units total  ({p50_daily:.1f}/day)\n"
        f"  Optimistic  (P90): {p90_total:,.0f} units total  ({p90.mean():.1f}/day)\n\n"
        f"Current Inventory:  {inv:,} units\n"
        f"Lead Time:          {lead_time} days\n"
        f"Safety Stock:       {safety_stock:.0f} units  (Z=1.65, σ={sigma:.1f})\n"
        f"Reorder Point:      {reorder_point:.0f} units\n"
        f"Recommendation:     {rec}"
    )


async def tool_query_mysql(sql: str, creds: dict | None = None) -> str:
    """
    Executes a read-only SQL query on the MySQL pet_store_scm database.
    creds (optional): {host, port, user, password, db} — overrides .env values.
    """
    sql_stripped = sql.strip()
    sql_lower = sql_stripped.lower()
    # Reject multi-statement queries — a semicolon anywhere after the first word
    # is a sign of statement stacking (e.g. SELECT 1; DROP TABLE ...)
    if ";" in sql_stripped.rstrip(";"):  # allow single trailing semicolon
        return "ERROR: Multi-statement queries are not allowed."
    # Strip leading block comments that could hide the real keyword
    import re as _re

    sql_uncommented = _re.sub(r"/\*.*?\*/", "", sql_lower, flags=_re.DOTALL).lstrip()
    if not any(
        sql_uncommented.startswith(kw)
        for kw in ("select", "show", "describe", "explain", "desc")
    ):
        return "ERROR: Only SELECT/SHOW/DESCRIBE queries are allowed."
    rows = await mysql_query(sql, creds=creds)
    if not rows:
        return "Query returned 0 rows."
    if rows and "error" in rows[0]:
        return f"MySQL Error: {rows[0]['error']}"
    serialized = json.dumps(rows, default=str, indent=2)
    if len(serialized) > 8000:
        serialized = serialized[:8000] + "\n... (truncated)"
    return f"MySQL Result ({len(rows)} rows):\n{serialized}"


async def tool_query_postgres(sql: str, creds: dict | None = None) -> str:
    """
    Executes a read-only SQL query on the PostgreSQL pet_store_scm database.
    creds (optional): {host, port, user, password, db} — overrides .env values.
    """
    sql_stripped = sql.strip()
    sql_lower = sql_stripped.lower()
    if ";" in sql_stripped.rstrip(";"):
        return "ERROR: Multi-statement queries are not allowed."
    import re as _re

    sql_uncommented = _re.sub(r"/\*.*?\*/", "", sql_lower, flags=_re.DOTALL).lstrip()
    if not any(
        sql_uncommented.startswith(kw)
        for kw in ("select", "show", "describe", "explain")
    ):
        return "ERROR: Only SELECT queries are allowed."
    rows = await pg_query(sql, creds=creds)
    if not rows:
        return "Query returned 0 rows."
    if rows and "error" in rows[0]:
        return f"PostgreSQL Error: {rows[0]['error']}"
    serialized = json.dumps(rows, default=str, indent=2)
    if len(serialized) > 8000:
        serialized = serialized[:8000] + "\n... (truncated)"
    return f"PostgreSQL Result ({len(rows)} rows):\n{serialized}"


async def tool_test_mysql_connection(creds: dict | None = None) -> str:
    """Tests the MySQL connection with the given credentials."""
    result = await test_mysql_connection(creds)
    if result["ok"]:
        d = result["details"]
        return (
            f"MySQL connection successful.\n"
            f"  Host:     {d.get('host')}:{d.get('port')}\n"
            f"  User:     {d.get('user', 'n/a')}\n"
            f"  Database: {d.get('database')}\n"
            f"  Version:  {d.get('version')}"
        )
    return f"MySQL connection failed: {result['message']}"


async def tool_test_postgres_connection(creds: dict | None = None) -> str:
    """Tests the PostgreSQL connection with the given credentials."""
    result = await test_postgres_connection(creds)
    if result["ok"]:
        d = result["details"]
        return (
            f"PostgreSQL connection successful.\n"
            f"  Host:     {d.get('host')}:{d.get('port')}\n"
            f"  Database: {d.get('database')}\n"
            f"  Version:  {d.get('version')}"
        )
    return f"PostgreSQL connection failed: {result['message']}"


def tool_get_supplier_info(supplier_name: str | None = None) -> str:
    """Returns structured supplier information from the knowledge base."""
    SUPPLIERS = {
        "PawsSupply Co": {
            "country": "United States",
            "on_time_pct": 96.2,
            "quality_rating": 4.7,
            "lead_time_days": "5-7",
            "min_order_qty": 50,
            "emergency_capable": True,
            "notes": "Primary distributor. Same-day emergency orders up to 500 units. Excellent reliability.",
            "categories": ["Dog Food", "Cat Food", "Health", "Accessories"],
        },
        "NutriPet Inc": {
            "country": "United States",
            "on_time_pct": 92.5,
            "quality_rating": 4.5,
            "lead_time_days": "8-12",
            "min_order_qty": 100,
            "emergency_capable": False,
            "notes": "Specialty nutrition. MOQ 100 units. Strong R&D pipeline for new SKUs.",
            "categories": ["Dog Food", "Cat Food"],
        },
        "GlobalPet Dist": {
            "country": "Mexico",
            "on_time_pct": 88.1,
            "quality_rating": 4.1,
            "lead_time_days": "14-21",
            "min_order_qty": 200,
            "emergency_capable": False,
            "notes": "Latin America distributor. Longer lead times; customs delays possible. Budget-friendly.",
            "categories": ["Dog Food", "Cat Food", "Accessories"],
        },
        "TreatWorld LLC": {
            "country": "China",
            "on_time_pct": 84.3,
            "quality_rating": 3.9,
            "lead_time_days": "18-25",
            "min_order_qty": 500,
            "emergency_capable": False,
            "notes": "Asia Pacific treats supplier. Quality incident Q4 2023 (batch TW-2023-Q4), resolved Q1 2024. Recommend dual-sourcing for critical treat SKUs.",
            "categories": ["Dog Treats", "Cat Treats"],
        },
        "HealthPet Labs": {
            "country": "United States",
            "on_time_pct": 94.8,
            "quality_rating": 4.6,
            "lead_time_days": "7-10",
            "min_order_qty": 50,
            "emergency_capable": True,
            "notes": "FDA registered facility. Pharmaceutical-grade supplements and flea/tick products.",
            "categories": ["Health"],
        },
        "VetPharma Supply": {
            "country": "Germany",
            "on_time_pct": 91.2,
            "quality_rating": 4.8,
            "lead_time_days": "12-18",
            "min_order_qty": 30,
            "emergency_capable": False,
            "notes": "EU-based vet pharma. Highest quality rating. Longer international shipping.",
            "categories": ["Health"],
        },
        "PetEssentials": {
            "country": "United States",
            "on_time_pct": 97.1,
            "quality_rating": 4.4,
            "lead_time_days": "5-7",
            "min_order_qty": 100,
            "emergency_capable": True,
            "notes": "Best on-time delivery rate. General accessories, litter, and consumables.",
            "categories": ["Accessories", "Cat Supplies"],
        },
        "ToyPet Factory": {
            "country": "China",
            "on_time_pct": 80.5,
            "quality_rating": 3.7,
            "lead_time_days": "18-25",
            "min_order_qty": 1000,
            "emergency_capable": False,
            "notes": "Budget toy supplier. Very high MOQ. Lead time variance ±5 days. Not suitable for critical stock.",
            "categories": ["Toys"],
        },
        "RawPet Foods": {
            "country": "United States",
            "on_time_pct": 95.0,
            "quality_rating": 4.9,
            "lead_time_days": "7-10",
            "min_order_qty": 20,
            "emergency_capable": True,
            "notes": "Premium raw/freeze-dried specialist. Cold chain logistics required. Highest quality overall.",
            "categories": ["Dog Food", "Cat Food"],
        },
    }

    if supplier_name:
        # fuzzy match
        match = next((s for s in SUPPLIERS if supplier_name.lower() in s.lower()), None)
        if not match:
            available = ", ".join(SUPPLIERS.keys())
            return f"Supplier '{supplier_name}' not found. Available: {available}"
        s = SUPPLIERS[match]
        return (
            f"=== Supplier Profile: {match} ===\n"
            f"Country:           {s['country']}\n"
            f"On-Time Delivery:  {s['on_time_pct']}%\n"
            f"Quality Rating:    {s['quality_rating']}/5.0\n"
            f"Lead Time:         {s['lead_time_days']} days\n"
            f"Min Order Qty:     {s['min_order_qty']} units\n"
            f"Emergency Orders:  {'Yes' if s['emergency_capable'] else 'No'}\n"
            f"Categories:        {', '.join(s['categories'])}\n"
            f"Notes:             {s['notes']}"
        )

    # summary table
    lines = ["=== All Supplier Performance Summary ===\n"]
    for name, s in SUPPLIERS.items():
        lines.append(
            f"{name:<20} | OTD: {s['on_time_pct']:>5}% | "
            f"Quality: {s['quality_rating']}/5.0 | "
            f"Lead: {s['lead_time_days']}d | "
            f"Emergency: {'Y' if s['emergency_capable'] else 'N'}"
        )
    return "\n".join(lines)


def tool_get_knowledge_base(topic: str) -> str:
    """
    Returns structured policy, guideline, or domain knowledge for a given topic.
    Topics: reorder_policy, safety_stock, flea_tick_seasonality, holiday_demand,
            supplier_risk, new_sku_ramp, cold_chain, regulatory, shrink_loss, kpi_targets
    """
    KB = {
        "reorder_policy": """
=== Reorder Policy – Pet Store Supply Chain ===

Standard Reorder Point Formula:
  Reorder Point = Average Daily Demand × Lead Time × Safety Factor (1.5)

Reorder Quantity (EOQ-based):
  Standard EOQ = √(2 × Annual Demand × Ordering Cost / Holding Cost)
  Simplified Default: 45-day supply for high-velocity SKUs, 60-day for slow movers.

By Category:
  Dog/Cat Food (Dry):     Reorder when ≤ 14 days of supply. Order 45-day supply.
  Dog/Cat Food (Wet):     Reorder when ≤ 10 days of supply. Order 30-day supply.
  Pet Health/Meds:        Reorder when ≤ 21 days of supply. Order 60-day supply.
  Treats:                 Reorder when ≤ 14 days of supply. Order 30-day supply.
  Accessories/Toys:       Reorder when ≤ 30 days of supply. Order 45-day supply.

Emergency Reorder Threshold: < 7 days of supply = CRITICAL, escalate immediately.
""",
        "safety_stock": """
=== Safety Stock Guidelines ===

Formula: Safety Stock = Z × σ_demand × √(lead_time)
  Z = 1.65 for 95% service level (standard)
  Z = 2.05 for 98% service level (high-value/critical SKUs)
  Z = 1.28 for 90% service level (slow-moving/low-margin SKUs)

Recommended Service Levels by Category:
  Dog/Cat Food (Dry):   95% SL  → Z=1.65
  Flea/Tick Products:   98% SL  → Z=2.05 (seasonal spike risk)
  Pet Medications:      98% SL  → Z=2.05 (health-critical)
  Treats/Accessories:   90% SL  → Z=1.28
  Cat Litter:           95% SL  → Z=1.65 (high-velocity, bulky)

Review safety stock monthly or after demand anomaly events.
""",
        "flea_tick_seasonality": """
=== Flea & Tick Seasonality Guidance ===

Peak Season: April – September (warm months)
  Demand uplift: +40-60% above baseline
  Pre-order window: 8-10 weeks before April 1

Key SKUs Affected: MED_001, MED_002, MED_003, MED_011

Recommended Action:
  - Build 90-day supply by March 15 each year
  - Coordinate with HealthPet Labs for priority allocation
  - VetPharma Supply (Germany) requires 12-week lead time for seasonal stock
  - Monitor weather forecasts: early warm spring → demand spikes 2-3 weeks earlier

Q2 2023 Incident: Flea/tick stockout in Southeast region due to missed seasonal build.
Loss: $280,000 in lost sales. Root cause: reorder trigger not adjusted for seasonality.
""",
        "holiday_demand": """
=== Holiday Demand Planning ===

Black Friday / Cyber Monday (late November):
  Toys/Accessories: +80-120% demand spike
  Treats: +60-90% demand spike
  Begin stock build by October 1

Christmas / New Year (December 20 – January 5):
  Dog/Cat Food Gift Packs: +40-60%
  Toys: +100-150%
  Accessories (collars, beds): +70-90%
  Begin stock build by November 1

Valentine's Day (February 14):
  Treats: +30-50% for one week
  Small toys/accessories: +20-30%

Supplier Notes:
  ToyPet Factory: Must be ordered 6-8 weeks ahead due to China factory closures
  during Chinese New Year (late Jan/early Feb)
  GlobalPet Dist: Add 1 week buffer for customs during holiday season
""",
        "supplier_risk": """
=== Supplier Risk Assessment ===

Risk Tiers:
  LOW RISK:    OTD ≥ 95%, Quality ≥ 4.5, Lead time < 10 days
  MEDIUM RISK: OTD 88-95%, Quality 4.0-4.5, or lead time 10-21 days
  HIGH RISK:   OTD < 88%, Quality < 4.0, or lead time > 21 days

Current Risk Classifications:
  LOW:    PawsSupply Co, PetEssentials, RawPet Foods, HealthPet Labs
  MEDIUM: NutriPet Inc, GlobalPet Dist, VetPharma Supply
  HIGH:   TreatWorld LLC (quality incident history), ToyPet Factory (OTD 80.5%)

Mitigation Strategies:
  - Never source >60% of a single category from one HIGH/MEDIUM risk supplier
  - Dual-source all CRITICAL and WARNING inventory SKUs
  - Maintain 2 approved alternate suppliers for all top-20 velocity SKUs
  - Quarterly supplier reviews with on-site audits for HIGH risk suppliers
""",
        "new_sku_ramp": """
=== New SKU Introduction Guidelines ===

Phase 1 – Pilot (Months 1-3):
  - Initial stock: 90-day supply based on comparable SKU demand
  - Reorder trigger: 45 days of supply (conservative)
  - Safety stock: Z=2.05 (high uncertainty buffer)

Phase 2 – Ramp (Months 4-9):
  - Recalculate demand baseline using Phase 1 actuals
  - Adjust reorder point and safety stock to standard levels
  - Switch to primary supplier contract

Phase 3 – Steady State (Month 10+):
  - Apply standard category reorder policy
  - Monitor for demand stabilization
""",
        "cold_chain": """
=== Cold Chain Requirements ===

Applicable SKUs: Freeze-dried and raw food lines (DOG_012, CAT_009)
Supplier: RawPet Foods (certified cold chain logistics partner)

Temperature Range: 32-40°F (0-4°C) throughout transit
Transit Time Maximum: 3 days ground, 1 day air freight
Packaging: Insulated EPS boxes with 48-hour gel packs

Receiving Protocol:
  - Inspect temperature log on arrival
  - Reject any shipment exceeding 45°F
  - Document and quarantine non-compliant lots
  - File claim with carrier within 24 hours of rejection

Storage:
  - Dedicated refrigerated section (40-50°F ambient)
  - FIFO rotation mandatory
  - Max shelf life: 18 months (frozen), 12 months (refrigerated)
""",
        "regulatory": """
=== Regulatory & Compliance Notes ===

FDA Compliance:
  - Pet food sold in the US must comply with AAFCO nutritional standards
  - All health/supplement SKUs from HealthPet Labs are FDA registered
  - VetPharma Supply (EU) products must have FDA import clearance

Recall Procedures:
  - Any supplier-notified quality recall: pull from shelves within 4 hours
  - Notify customers who purchased in last 30 days via email
  - Log all recalled units in the reorder_events table (status=CANCELLED)
  - Document incident in demand_anomalies table

Past Incidents:
  - TreatWorld LLC batch TW-2023-Q4: Elevated moisture content, 
    potential mold risk. 2,400 units recalled. Resolved Q1 2024.
""",
        "shrink_loss": """
=== Shrink & Loss Management ===

Target Shrink Rate: <1.5% of inventory value annually
Current Benchmark: 1.8% (above target, improvement needed)

Main Shrink Sources:
  1. Expired product (pet food): 42% of shrink
  2. Damage in transit: 28%
  3. Theft/administrative error: 20%
  4. Recall/quality holds: 10%

Mitigation:
  - Implement FIFO strictly for all food SKUs
  - Flag inventory approaching 75% of shelf life for markdown/promotion
  - Weekly cycle counts for high-value SKUs (MED_001-MED_015, DOG_001-DOG_006)
""",
        "kpi_targets": """
=== Supply Chain KPI Targets – Pet Store ===

Inventory & Service:
  In-Stock Rate (ISR):           ≥ 97.5%
  Stockout Rate:                 ≤ 2.5%
  Days of Supply (avg):          35-45 days
  Inventory Turnover:            8-12× per year

Forecast Accuracy:
  MAPE (demand forecast):        ≤ 15%
  Forecast Bias:                 ±5%
  Forecast Horizon:              30-day primary, 90-day strategic

Supplier:
  Overall On-Time Delivery:      ≥ 92%
  Defect Rate:                   ≤ 1.0%
  Fill Rate:                     ≥ 97%

Financial:
  Carrying Cost of Inventory:    ≤ 25% of inventory value/year
  Order Processing Cost:         ≤ $18 per purchase order
  Shrink Rate:                   ≤ 1.5%
""",
    }

    topic_lower = topic.lower().replace(" ", "_").replace("-", "_")
    # fuzzy match
    match = next((k for k in KB if topic_lower in k or k in topic_lower), None)
    if match:
        return KB[match].strip()

    available = ", ".join(KB.keys())
    # Try keyword search
    results = []
    for key, content in KB.items():
        if any(word in content.lower() for word in topic_lower.split("_")):
            results.append(f"[{key}]\n{content[:400]}...")
    if results:
        return f"Keyword search results for '{topic}':\n\n" + "\n\n---\n\n".join(
            results
        )

    return (
        f"Topic '{topic}' not found in knowledge base.\n"
        f"Available topics: {available}\n"
        "Try keywords like: reorder, safety stock, flea, holiday, supplier, cold chain, recall, kpi"
    )


async def tool_log_forecast_to_postgres(
    sku_id: str,
    p10_total: float,
    p50_total: float,
    p90_total: float,
    p50_daily: float,
    horizon_days: int = 30,
    forecast_source: str = "TFT",
    model_version: str = "v1.0",
) -> str:
    # forecast_date is NOT NULL in the schema — supply today's date
    sql = """
        INSERT INTO sku_forecasts
            (forecast_date, sku_id, horizon_days, p10_total, p50_total, p90_total,
             p50_daily, forecast_source, model_version)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (forecast_date, sku_id, horizon_days, forecast_source) DO NOTHING
    """
    result = await pg_execute(
        sql,
        (
            date.today(),
            sku_id.upper(),
            horizon_days,
            p10_total,
            p50_total,
            p90_total,
            p50_daily,
            forecast_source,
            model_version,
        ),
    )
    return f"Forecast logged to PostgreSQL for {sku_id}: {result}"


async def tool_create_inventory_alert(
    sku_id: str,
    alert_type: str,
    days_of_supply: float,
    current_inventory: int,
    avg_daily_demand: float,
    lead_time_days: int,
    recommended_action: str,
) -> str:
    sql = """
        INSERT INTO inventory_alerts
            (sku_id, alert_type, days_of_supply, current_inventory,
             avg_daily_demand, lead_time_days, recommended_action)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
    """
    result = await pg_execute(
        sql,
        (
            sku_id.upper(),
            alert_type.upper(),
            days_of_supply,
            current_inventory,
            avg_daily_demand,
            lead_time_days,
            recommended_action,
        ),
    )
    return f"Alert created in PostgreSQL for {sku_id} ({alert_type}): {result}"


async def tool_get_active_alerts(limit: int = 20) -> str:
    rows = await pg_query(
        "SELECT * FROM active_alerts ORDER BY alert_date DESC LIMIT $1",
        (limit,),
    )
    if not rows:
        return "No active alerts in PostgreSQL."
    if "error" in rows[0]:
        return f"PostgreSQL Error: {rows[0]['error']}"
    return f"Active Alerts ({len(rows)}):\n" + json.dumps(rows, default=str, indent=2)


async def tool_get_monthly_kpis(sku_id: str | None = None, months: int = 6) -> str:
    if sku_id:
        sql = """
            SELECT * FROM monthly_kpis
            WHERE sku_id = $1
            ORDER BY year_month DESC LIMIT $2
        """
        rows = await pg_query(sql, (sku_id.upper(), months))
    else:
        # BUG-006 fix: COALESCE handles both setup.py schema (revenue_est_usd)
        # and migrate_huft.py schema (revenue_est_inr)
        sql = """
            SELECT year_month,
                   SUM(total_demand) AS total_demand,
                   AVG(fill_rate_pct) AS avg_fill_rate_pct,
                   SUM(stockout_days) AS total_stockout_days,
                   SUM(COALESCE(revenue_est_inr, revenue_est_usd, 0)) AS total_revenue_inr
            FROM monthly_kpis
            GROUP BY year_month
            ORDER BY year_month DESC LIMIT $1
        """
        rows = await pg_query(sql, (months,))
    if not rows:
        return "No KPI data in PostgreSQL. Run the monthly KPI aggregation job first."
    if "error" in rows[0]:
        return f"PostgreSQL Error: {rows[0]['error']}"
    return f"Monthly KPIs:\n" + json.dumps(rows, default=str, indent=2)


# ── Web Search Tool ───────────────────────────────────────────────────────────


def tool_web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using Google (via SerpAPI) with automatic fallback to
    DuckDuckGo when no SERPAPI_KEY is configured.

    Returns a structured summary of the top results including titles,
    URLs, and snippets — ready for the agent to reason over.
    """
    serpapi_key = os.getenv("SERPAPI_KEY", "").strip()

    # ── Google via SerpAPI (preferred) ────────────────────────────────────
    if serpapi_key:
        try:
            from serpapi import GoogleSearch

            params = {
                "q": query,
                "api_key": serpapi_key,
                "num": min(num_results, 10),
                "hl": "en",
                "gl": "us",
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            organic = results.get("organic_results", [])
            if not organic:
                return f"No Google results found for: {query}"

            lines = [f"Google Search Results for: '{query}'\n"]
            for i, r in enumerate(organic[:num_results], 1):
                lines.append(f"{i}. {r.get('title', 'No title')}")
                lines.append(f"   URL: {r.get('link', '')}")
                lines.append(f"   {r.get('snippet', 'No description')}\n")

            # Include knowledge panel / answer box if present
            if "answer_box" in results:
                ab = results["answer_box"]
                answer = ab.get("answer") or ab.get("snippet", "")
                if answer:
                    lines.insert(1, f"Direct Answer: {answer}\n")

            return "\n".join(lines)

        except Exception as exc:
            logger.warning(
                f"[WebSearch] SerpAPI failed: {exc} — falling back to DuckDuckGo"
            )

    # ── DuckDuckGo fallback (no API key needed) ───────────────────────────
    try:
        from ddgs import DDGS

        lines = [f"Web Search Results for: '{query}'\n"]
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(num_results, 10)))
        if not results:
            return f"No results found for: {query}"
        for i, r in enumerate(results[:num_results], 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   URL: {r.get('href', '')}")
            lines.append(f"   {r.get('body', 'No description')}\n")
        return "\n".join(lines)

    except Exception as exc:
        return (
            f"Web search failed: {exc}\n"
            "Set SERPAPI_KEY in .env for Google Search, "
            "or install duckduckgo-search for free fallback."
        )


# ── Python REPL Tool ──────────────────────────────────────────────────────────


def tool_python_repl(code: str) -> str:
    """
    Execute arbitrary Python code in a secure, sandboxed namespace and return
    stdout + the value of the last expression.

    The execution context pre-loads:
      - pandas as pd
      - numpy as np
      - the full supply-chain DataFrame as `df`  (from CSV)
      - datetime, math, json, re, collections

    Dangerous built-ins (open, exec, eval, __import__, os, sys, subprocess)
    are blocked. Each call gets a fresh namespace — no state persists between calls.

    Returns the captured stdout output and/or the repr of the last expression.
    Use this for:
      - Data quality checks  (df[df['demand'] < 0])
      - Statistical analysis (df.groupby('category')['demand'].describe())
      - Custom calculations  (e.g. re-order quantities, MAPE computation)
      - Anything SQL cannot express easily
    """
    import ast
    import io
    import math
    import re
    import traceback as _tb
    from contextlib import redirect_stdout

    # ── Safe built-ins whitelist ──────────────────────────────────────────
    # SECURITY: Removed sandbox-escape vectors:
    #   getattr / setattr  → allows [].__class__.__base__.__subclasses__() chain
    #   dir / vars         → enumerates __dict__ / module internals
    #   type               → constructs new classes, accesses __mro__
    #   object             → base class gives access to all subclasses
    #   issubclass         → subclass enumeration
    # Also block via AST: any attribute access whose name starts with "__"
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hasattr": hasattr,
        "hash": hash,
        "int": int,
        "isinstance": isinstance,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }

    # ── Build execution namespace ─────────────────────────────────────────
    import numpy as np
    import pandas as pd
    import json as _json
    import collections as _collections
    import datetime as _datetime

    namespace: dict = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "math": math,
        "re": re,
        "json": _json,
        "collections": _collections,
        "datetime": _datetime,
        "df": get_df().copy(),  # full supply-chain DataFrame
    }

    # ── Parse and validate code ───────────────────────────────────────────
    _BLOCKED_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "pickle",
        "shelve",
        "ctypes",
        "multiprocessing",
        "threading",
        "signal",
        "pathlib",
        "importlib",
        "builtins",
        "inspect",
        "gc",
        "weakref",
        "atexit",
        "platform",
        "resource",
        "pwd",
        "grp",
        "termios",
        "tty",
    }
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Block dangerous imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    root = name.split(".")[0]
                    if root in _BLOCKED_IMPORTS:
                        return f"SecurityError: import of '{root}' is not allowed."
            # Block ALL dunder attribute access (e.g. __class__, __subclasses__)
            # This closes the getattr() sandbox-escape chain even if getattr were present
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return (
                        f"SecurityError: access to dunder attribute "
                        f"'{node.attr}' is not allowed in the sandbox."
                    )
            # Block names that reference dunder globals
            if isinstance(node, ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    return (
                        f"SecurityError: reference to '{node.id}' "
                        f"is not allowed in the sandbox."
                    )
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    # ── Execute with captured stdout ──────────────────────────────────────
    stdout_buf = io.StringIO()
    last_value = None
    try:
        # Split into statements; eval the last expression for its value
        statements = list(ast.iter_child_nodes(tree))
        if not statements:
            return "No code to execute."

        compile_all = compile(tree, "<repl>", "exec")
        with redirect_stdout(stdout_buf):
            exec(compile_all, namespace)  # noqa: S102

        # Try to get the value of the last expression
        last_node = statements[-1]
        if isinstance(last_node, ast.Expr):
            try:
                last_value = eval(  # noqa: S307
                    compile(ast.Expression(last_node.value), "<repl>", "eval"),
                    namespace,
                )
            except Exception:
                pass

    except Exception:
        tb = _tb.format_exc(limit=5)
        output = stdout_buf.getvalue()
        return f"Execution error:\n{tb}" + (
            f"\nOutput before error:\n{output}" if output else ""
        )

    output = stdout_buf.getvalue()
    parts = []
    if output:
        parts.append(output.rstrip())
    if last_value is not None:
        # Truncate very large repr (e.g. full DataFrames)
        r = repr(last_value)
        if len(r) > 3000:
            r = r[:3000] + f"\n... [{len(r) - 3000} more chars truncated]"
        parts.append(r)
    return "\n".join(parts) if parts else "Code executed successfully (no output)."


# ── Data Quality & Profiling Tool ─────────────────────────────────────────────


def tool_data_quality(
    table: str = "all",
    checks: str = "all",
) -> str:
    """
    Run a comprehensive data quality and statistical profiling audit on the
    supply-chain dataset.

    table  : "daily_demand" | "skus" | "all"
    checks : "all" | "negatives" | "nulls" | "outliers" | "profile" | "anomalies"

    Returns a structured report covering:
      - Row / column counts
      - Null / missing value counts per column
      - Negative values in numeric columns that should always be positive
      - Statistical outliers (values beyond mean ± 3σ per SKU)
      - Demand anomalies (days where demand > 3× 30-day rolling mean)
      - Inventory anomalies (stock drops of > 80% in a single day)
      - Data freshness (date range, most recent record)
      - Duplicate detection
      - Value distribution summary per key column
    """
    df = get_df()
    if df is None or df.empty:
        return "No data loaded. CSV not found or empty."

    run_all = checks == "all"
    lines: list[str] = ["╔══ Data Quality & Profiling Report ══╗\n"]

    # ── Filter to requested table ─────────────────────────────────────────
    demand_cols = [
        "sku_id",
        "date",
        "demand",
        "inventory",
        "lead_time_days",
        "price_usd",
    ]
    sku_static_cols = [
        "sku_id",
        "category",
        "subcategory",
        "supplier",
        "region",
        "lead_time_days",
        "price_usd",
    ]

    if table == "daily_demand":
        frames = {
            "daily_demand": df[demand_cols]
            if all(c in df.columns for c in demand_cols)
            else df
        }
    elif table == "skus":
        avail = [c for c in sku_static_cols if c in df.columns]
        frames = {"skus": df[avail].drop_duplicates("sku_id")}
    else:
        avail_demand = [c for c in demand_cols if c in df.columns]
        avail_sku = [c for c in sku_static_cols if c in df.columns]
        frames = {
            "daily_demand": df[avail_demand],
            "skus": df[avail_sku].drop_duplicates("sku_id"),
        }

    for tname, tdf in frames.items():
        lines.append(f"── Table: {tname} ──")
        lines.append(f"  Rows: {len(tdf):,}  |  Columns: {len(tdf.columns)}")

    # ── Profile ───────────────────────────────────────────────────────────
    if run_all or "profile" in checks:
        lines.append("\n── Statistical Profile (demand column) ──")
        if "demand" in df.columns:
            d = df["demand"]
            lines.append(f"  Mean:    {d.mean():.2f}")
            lines.append(f"  Median:  {d.median():.2f}")
            lines.append(f"  Std Dev: {d.std():.2f}")
            lines.append(f"  Min:     {d.min():.2f}")
            lines.append(f"  Max:     {d.max():.2f}")
            lines.append(f"  Zeros:   {(d == 0).sum():,}")

        if "inventory" in df.columns:
            inv = df["inventory"]
            lines.append(
                f"\n  Inventory — Mean: {inv.mean():.0f}  "
                f"Min: {inv.min():.0f}  Max: {inv.max():.0f}  "
                f"Zeros: {(inv == 0).sum():,}"
            )

        lines.append(
            f"\n  Date range: {df['date'].min().date()} → {df['date'].max().date()}"
            if "date" in df.columns
            else ""
        )
        lines.append(
            f"  SKUs: {df['sku_id'].nunique()}" if "sku_id" in df.columns else ""
        )
        lines.append(
            f"  Categories: {df['category'].nunique()}"
            if "category" in df.columns
            else ""
        )

    # ── Nulls ─────────────────────────────────────────────────────────────
    if run_all or "nulls" in checks:
        lines.append("\n── Null / Missing Values ──")
        null_counts = df.isnull().sum()
        found_nulls = null_counts[null_counts > 0]
        if found_nulls.empty:
            lines.append("  No null values found.")
        else:
            for col, cnt in found_nulls.items():
                pct = cnt / len(df) * 100
                lines.append(f"  {col}: {cnt:,} nulls ({pct:.1f}%)")

    # ── Negatives ─────────────────────────────────────────────────────────
    if run_all or "negatives" in checks:
        lines.append("\n── Negative Values (should always be ≥ 0) ──")
        check_cols = ["demand", "inventory", "lead_time_days", "price_usd"]
        found_any = False
        for col in check_cols:
            if col not in df.columns:
                continue
            neg = df[df[col] < 0]
            if not neg.empty:
                found_any = True
                sample_skus = (
                    neg["sku_id"].unique()[:5].tolist()
                    if "sku_id" in neg.columns
                    else []
                )
                lines.append(
                    f"  {col}: {len(neg):,} negative rows — "
                    f"min={df[col].min():.2f}  "
                    f"affected SKUs: {sample_skus}"
                )
        if not found_any:
            lines.append(
                "  No negative values found in demand, inventory, price, or lead time."
            )

    # ── Outliers (per-SKU z-score) ────────────────────────────────────────
    if run_all or "outliers" in checks:
        lines.append("\n── Demand Outliers (|z-score| > 3 per SKU) ──")
        if "demand" in df.columns and "sku_id" in df.columns:
            grp = df.groupby("sku_id")["demand"]
            mu = grp.transform("mean")
            sigma = grp.transform("std").fillna(1)
            z = (df["demand"] - mu) / sigma
            outliers = df[z.abs() > 3]
            if outliers.empty:
                lines.append("  No per-SKU outliers found (|z| > 3).")
            else:
                lines.append(
                    f"  {len(outliers):,} outlier rows across "
                    f"{outliers['sku_id'].nunique()} SKUs."
                )
                worst = outliers.assign(z=z[z.abs() > 3].abs()).nlargest(5, "z")[
                    ["sku_id", "date", "demand", "z"]
                ]
                for _, row in worst.iterrows():
                    lines.append(
                        f"    SKU {row['sku_id']}  date={str(row['date'])[:10]}  "
                        f"demand={row['demand']:.0f}  z={row['z']:.1f}"
                    )

    # ── Anomalies ─────────────────────────────────────────────────────────
    if run_all or "anomalies" in checks:
        lines.append("\n── Demand Spike Anomalies (> 3× 30-day rolling mean) ──")
        if "demand" in df.columns and "sku_id" in df.columns and "date" in df.columns:
            df_s = df.sort_values(["sku_id", "date"])
            roll = df_s.groupby("sku_id")["demand"].transform(
                lambda x: x.rolling(30, min_periods=5).mean().shift(1)
            )
            spikes = df_s[df_s["demand"] > roll * 3].dropna(subset=["demand"])
            if spikes.empty:
                lines.append("  No demand spikes detected.")
            else:
                lines.append(
                    f"  {len(spikes):,} spike rows across "
                    f"{spikes['sku_id'].nunique()} SKUs."
                )
                for _, row in spikes.nlargest(5, "demand")[
                    ["sku_id", "date", "demand"]
                ].iterrows():
                    lines.append(
                        f"    SKU {row['sku_id']}  {str(row['date'])[:10]}  "
                        f"demand={row['demand']:.0f}"
                    )

        lines.append("\n── Inventory Drop Anomalies (single-day drop > 80%) ──")
        if "inventory" in df.columns and "sku_id" in df.columns:
            df_s2 = df.sort_values(["sku_id", "date"])
            prev_inv = df_s2.groupby("sku_id")["inventory"].shift(1)
            pct_drop = (prev_inv - df_s2["inventory"]) / (prev_inv + 1)
            big_drops = df_s2[pct_drop > 0.8]
            if big_drops.empty:
                lines.append("  No sudden inventory drops detected.")
            else:
                lines.append(
                    f"  {len(big_drops):,} sudden drop rows across "
                    f"{big_drops['sku_id'].nunique()} SKUs."
                )

    # ── Duplicates ────────────────────────────────────────────────────────
    if run_all or "profile" in checks:
        lines.append("\n── Duplicate Detection ──")
        if "sku_id" in df.columns and "date" in df.columns:
            dups = df.duplicated(subset=["sku_id", "date"]).sum()
            lines.append(
                f"  (sku_id, date) duplicates: {dups:,}"
                + (" — CLEAN" if dups == 0 else " — REVIEW NEEDED")
            )

    lines.append("\n╚══ End of Report ══╝")
    return "\n".join(str(l) for l in lines)


# ── NEW TOOLS (20) ───────────────────────────────────────────────────────────


def tool_get_brand_performance(brand: str | None = None, top_n: int = 10) -> str:
    """Ranks brands by revenue from sales transactions."""
    try:
        txn = get_transactions()
        prod = get_products()
        ret = get_returns()
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    # Revenue & units by brand
    brand_rev = (
        txn.groupby("brand")
        .agg(
            total_revenue_inr=("net_revenue_inr", "sum"),
            total_units_sold=("quantity", "sum"),
            total_margin_inr=("gross_margin_inr", "sum"),
        )
        .reset_index()
    )
    brand_rev["avg_margin_pct"] = (
        brand_rev["total_margin_inr"]
        / brand_rev["total_revenue_inr"].replace(0, np.nan)
        * 100
    ).round(1)

    # Return rate per brand
    total_sold = (
        txn.groupby("brand")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "sold"})
    )
    total_ret = (
        ret.groupby("brand")["quantity_returned"]
        .sum()
        .reset_index()
        .rename(columns={"quantity_returned": "returned"})
    )
    rr = total_sold.merge(total_ret, on="brand", how="left")
    rr["returned"] = rr["returned"].fillna(0)
    rr["return_rate_pct"] = (
        rr["returned"] / rr["sold"].replace(0, np.nan) * 100
    ).round(2)

    # Stockout days (demand > 0 but inventory == 0)
    dem_copy = dem.copy()
    dem_copy["stockout"] = (dem_copy["inventory"] == 0) & (dem_copy["demand"] > 0)
    stockout = (
        dem_copy.groupby("brand")["stockout"]
        .sum()
        .reset_index()
        .rename(columns={"stockout": "stockout_days"})
    )

    result = brand_rev.merge(rr[["brand", "return_rate_pct"]], on="brand", how="left")
    result = result.merge(stockout, on="brand", how="left")
    result["return_rate_pct"] = result["return_rate_pct"].fillna(0)
    result["stockout_days"] = result["stockout_days"].fillna(0).astype(int)
    result = result.sort_values("total_revenue_inr", ascending=False)

    if brand:
        row = result[result["brand"].str.contains(brand, case=False, na=False)]
        if row.empty:
            return f"Brand '{brand}' not found. Available: {', '.join(result['brand'].head(10))}"
        r = row.iloc[0]
        return (
            f"=== Brand Performance: {r['brand']} ===\n"
            f"Total Revenue       : ₹{r['total_revenue_inr']:,.0f}\n"
            f"Total Units Sold    : {int(r['total_units_sold']):,}\n"
            f"Avg Margin          : {r['avg_margin_pct']:.1f}%\n"
            f"Return Rate         : {r['return_rate_pct']:.2f}%\n"
            f"Stockout Days       : {r['stockout_days']:,}\n"
        )

    lines = [f"=== Top {min(top_n, len(result))} Brands by Revenue ===\n"]
    for rank, (_, r) in enumerate(result.head(top_n).iterrows(), 1):
        flag = " ⚠ HIGH RETURNS" if r["return_rate_pct"] > 5 else ""
        lines.append(
            f"#{rank} {r['brand']}\n"
            f"   Revenue: ₹{r['total_revenue_inr']:,.0f} | Units: {int(r['total_units_sold']):,} | "
            f"Margin: {r['avg_margin_pct']:.1f}% | Returns: {r['return_rate_pct']:.2f}% | "
            f"Stockout Days: {r['stockout_days']}{flag}\n"
        )
    return "\n".join(lines)


def tool_get_franchise_inventory_comparison(
    region: str | None = None, store_type: str | None = None
) -> str:
    """
    Compare inventory health across stores/regions.

    Uses store_daily_inventory.csv (per-store, per-SKU data) to return a
    Markdown table of stores ranked by risk, with store-level detail.
    Falls back to category-level aggregation if the per-store CSV is unavailable.
    """
    try:
        stores = get_stores()
        sdi_path = DATA_DIR / "store_daily_inventory.csv"

        # ── Primary path: per-store data from store_daily_inventory.csv ──────
        if sdi_path.exists():
            sdi = pd.read_csv(sdi_path, parse_dates=["date"])
            latest_date = sdi["date"].max()
            sdi = sdi[sdi["date"] == latest_date].copy()

            if region:
                sdi = sdi[sdi["region"].str.contains(region, case=False, na=False)]
            if store_type and "store_type" in sdi.columns:
                sdi = sdi[
                    sdi["store_type"].str.contains(store_type, case=False, na=False)
                ]

            # Per-store summary: critical SKU count, worst DoS, total SKUs
            store_summary = (
                sdi.groupby(["store_id", "city", "state", "region"])
                .agg(
                    total_skus=("sku_id", "nunique"),
                    critical_skus=("risk_status", lambda x: (x == "CRITICAL").sum()),
                    warning_skus=("risk_status", lambda x: (x == "WARNING").sum()),
                    min_dos=("days_of_supply", "min"),
                    avg_dos=("days_of_supply", "mean"),
                )
                .reset_index()
                .sort_values(["critical_skus", "min_dos"], ascending=[False, True])
            )

            scope = f"{region} region" if region else "all regions"
            header = (
                f"## Franchise Inventory Comparison — {scope} (as of {latest_date.date()})\n\n"
                f"**{len(store_summary)} stores** · "
                f"🔴 {store_summary['critical_skus'].gt(0).sum()} stores with critical SKUs · "
                f"🟡 {store_summary['warning_skus'].gt(0).sum()} stores with warnings\n\n"
            )

            tbl = (
                "| Store ID | City | State | Region | Critical SKUs | Warning SKUs "
                "| Worst DoS | Avg DoS | Total SKUs |\n"
                "|----------|------|-------|--------|--------------|-------------|"
                "----------|---------|------------|\n"
            )
            for _, r in store_summary.iterrows():
                crit_flag = "🔴" if int(r["critical_skus"]) > 0 else "✅"
                tbl += (
                    f"| {r['store_id']} | {r['city']} | {r['state']} | {r['region']} "
                    f"| {crit_flag} {int(r['critical_skus'])} "
                    f"| 🟡 {int(r['warning_skus'])} "
                    f"| {r['min_dos']:.1f}d | {r['avg_dos']:.1f}d "
                    f"| {int(r['total_skus'])} |\n"
                )

            # Top critical SKUs across all stores
            critical_skus = (
                sdi[sdi["risk_status"] == "CRITICAL"]
                .groupby(["sku_id", "name", "category"])
                .agg(
                    affected_stores=("store_id", "nunique"),
                    min_dos=("days_of_supply", "min"),
                )
                .reset_index()
                .sort_values("affected_stores", ascending=False)
                .head(10)
            )

            sku_section = ""
            if not critical_skus.empty:
                sku_section = "\n### Top Critical SKUs (appearing in most stores)\n\n"
                sku_section += (
                    "| SKU | Product | Category | Stores Affected | Worst DoS |\n"
                    "|-----|---------|----------|----------------|----------|\n"
                )
                for _, r in critical_skus.iterrows():
                    sku_section += (
                        f"| {r['sku_id']} | {str(r['name'])[:30]} | {r['category']} "
                        f"| **{int(r['affected_stores'])}** | {r['min_dos']:.1f}d |\n"
                    )

            rec = (
                "\n**Next steps:**\n"
                "1. Run `get_store_inventory_breakdown` with `risk_status=CRITICAL` for full per-SKU detail.\n"
                "2. Run `generate_purchase_order` to create supplier POs for the critical SKUs above.\n"
                "3. Consider stock transfers from high-DoS stores to critical ones.\n"
            )
            return header + tbl + sku_section + rec

        # ── Fallback path: category-level aggregation from demand CSV ────────
        dem = get_df()
        latest_date = dem["date"].max()
        latest = dem[dem["date"] == latest_date].copy()
        cutoff = latest_date - pd.Timedelta(days=30)
        avg_demand = (
            dem[dem["date"] >= cutoff]
            .groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "avg_daily_demand"})
        )
        latest = latest.merge(avg_demand, on="sku_id", how="left")
        latest["days_of_supply"] = (
            latest["inventory"] / latest["avg_daily_demand"].replace(0, np.nan)
        ).fillna(0)
        latest["critical"] = latest["days_of_supply"] < latest["lead_time_days"]

        cat_inv = (
            latest.groupby("category")
            .agg(
                avg_dos=("days_of_supply", "mean"),
                critical_skus=("critical", "sum"),
                total_skus=("sku_id", "count"),
            )
            .reset_index()
            .sort_values("avg_dos")
        )

        header = f"## Franchise Inventory Comparison — Category Level (as of {latest_date.date()})\n\n"
        header += "*(Note: store_daily_inventory.csv not found — showing category aggregates)*\n\n"
        tbl = (
            "| Category | Avg Days of Supply | Critical SKUs | Total SKUs |\n"
            "|----------|-------------------|--------------|------------|\n"
        )
        for _, r in cat_inv.iterrows():
            flag = "🔴 " if int(r["critical_skus"]) > 0 else "✅ "
            tbl += f"| {r['category']} | {r['avg_dos']:.1f}d | {flag}{int(r['critical_skus'])} | {int(r['total_skus'])} |\n"

        at_risk = latest[latest["critical"]].sort_values("days_of_supply").head(10)
        sku_section = ""
        if not at_risk.empty:
            sku_section = "\n### Critical SKUs\n\n| SKU | Product | Category | Inventory | DoS | Lead Time |\n|-----|---------|----------|-----------|-----|-----------|\n"
            for _, s in at_risk.iterrows():
                sku_section += f"| {s['sku_id']} | {str(s['name'])[:30]} | {s['category']} | {int(s['inventory']):,} | {s['days_of_supply']:.1f}d | {int(s['lead_time_days'])}d |\n"

        return header + tbl + sku_section

    except Exception as e:
        return f"Error in franchise comparison: {e}"


def tool_get_seasonal_demand_calendar(
    category: str | None = None, months_ahead: int = 3
) -> str:
    """Compute monthly demand indices and overlay with Indian festival calendar."""
    try:
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    dem = dem.copy()
    dem["month"] = dem["date"].dt.month
    dem["year"] = dem["date"].dt.year

    if category:
        filtered = dem[dem["category"].str.contains(category, case=False, na=False)]
        if filtered.empty:
            return f"No data for category '{category}'."
        dem = filtered

    # Monthly demand index: monthly avg / annual avg per category
    cat_annual = dem.groupby("category")["demand"].mean().rename("annual_avg")
    monthly_cat = (
        dem.groupby(["category", "month"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "monthly_avg"})
    )
    monthly_cat = monthly_cat.merge(cat_annual, on="category")
    monthly_cat["demand_index"] = (
        monthly_cat["monthly_avg"] / monthly_cat["annual_avg"] * 100
    ).round(1)

    # Indian festival / seasonal calendar
    FESTIVALS = {
        1: ["Republic Day (Jan 26) — premium accessories spike +20%"],
        3: [
            "Holi (Mar) — toys/accessories +30%",
            "Spring — flea/tick pre-stock season begins",
        ],
        6: ["Monsoon begins — tick/flea +40%, grooming products +25%"],
        7: ["Monsoon peak — waterproof accessories, health products"],
        8: ["Independence Day (Aug 15) — promotions opportunity"],
        9: ["Monsoon end — inventory normalisation"],
        10: [
            "Navratri (Oct) — premium food gifts +35%",
            "Dussehra — accessories +30%",
            "Diwali prep — all categories +50%",
        ],
        11: ["Diwali (Oct/Nov) — peak demand all categories +50-80%"],
        12: ["Christmas/New Year — premium products, gifts +40%"],
    }

    from datetime import date as dt_date

    today = pd.Timestamp.today()
    current_month = today.month

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    lines = [
        f"=== Seasonal Demand Calendar — {'All Categories' if not category else category} ===\n"
    ]
    lines.append("── Monthly Demand Indices (100 = annual average) ──\n")

    # Show upcoming months
    lines.append(
        f"Upcoming {months_ahead} months (starting from current month {month_names[current_month - 1]}):\n"
    )
    for i in range(months_ahead):
        m = ((current_month - 1 + i) % 12) + 1
        m_name = month_names[m - 1]
        cat_slice = (
            monthly_cat[monthly_cat["month"] == m]
            if not category
            else monthly_cat[
                (monthly_cat["month"] == m)
                & (monthly_cat["category"].str.contains(category, case=False, na=False))
            ]
        )
        festivals = FESTIVALS.get(m, [])
        lines.append(f"  {m_name} (Month {m}):")
        for _, row in (
            cat_slice.sort_values("demand_index", ascending=False).head(5).iterrows()
        ):
            bar = "█" * int(row["demand_index"] / 20)
            lines.append(
                f"    {row['category']:25s} index={row['demand_index']:5.1f}  {bar}"
            )
        for fest in festivals:
            lines.append(f"    [FESTIVAL] {fest}")
        lines.append("")

    # Pre-stock recommendations
    high_idx = monthly_cat[monthly_cat["demand_index"] > 120].sort_values(
        "demand_index", ascending=False
    )
    if not high_idx.empty:
        lines.append("── Pre-Stock Recommendations (demand index > 120) ──")
        for _, r in high_idx.head(8).iterrows():
            m_name = month_names[int(r["month"]) - 1]
            lines.append(
                f"  Build stock 4-6 weeks before {m_name}: "
                f"{r['category']} (index={r['demand_index']:.0f}, "
                f"+{r['demand_index'] - 100:.0f}% above baseline)"
            )

    return "\n".join(lines)


def tool_get_cold_chain_monitor(days_ahead: int = 7) -> str:
    """Monitor cold chain status: temperature breaches, expiry risks, waste value."""
    try:
        cc = get_cold_chain()
    except FileNotFoundError as e:
        return f"Cold chain data not available: {e}"

    # BUG-043 fix: use dataset's own max date (not system today) so breach
    # detection works even when the data is historical (e.g., 2023-2024 CSV
    # running on a server in 2026 would otherwise always show 0 breaches).
    data_latest = cc["date"].max()
    recent_cutoff = data_latest - pd.Timedelta(days=7)
    recent = cc[cc["date"] >= recent_cutoff]
    breaches = recent[recent["temp_breach"].astype(str).str.lower() == "true"]

    # Units at risk of expiry in next days_ahead days (relative to data end)
    expiry_cutoff = data_latest + pd.Timedelta(days=days_ahead)
    at_expiry = cc[
        (cc["expiry_date"] <= expiry_cutoff) & (cc["expiry_date"] >= data_latest)
    ].copy()

    # Critical: shelf life < 3 days
    critical_expiry = cc[cc["shelf_life_days_remaining"] < 3]

    # Estimate waste value
    prod = None
    try:
        prod = get_products()
        if not at_expiry.empty:
            at_expiry = at_expiry.merge(
                prod[["sku_id", "cost_inr"]], on="sku_id", how="left"
            )
            at_expiry["waste_value_inr"] = at_expiry[
                "units_at_risk_of_expiry"
            ] * at_expiry["cost_inr"].fillna(0)
    except Exception:
        pass

    total_waste = (
        at_expiry["waste_value_inr"].sum()
        if "waste_value_inr" in at_expiry.columns
        else 0
    )

    # BUG-16 fix: 'today' was never defined in this function; use data_latest
    today = data_latest
    lines = [f"=== Cold Chain Monitor — {today.date()} (data as of) ===\n"]

    lines.append(f"── Temperature Breaches (last 7 days) ──")
    if breaches.empty:
        lines.append("  No temperature breaches detected.")
    else:
        lines.append(f"  *** {len(breaches)} breach events detected! ***")
        for _, b in breaches.head(10).iterrows():
            lines.append(
                f"  [{b['date'].date()}] {b['sku_id']} – {b['name']}: "
                f"{b['temp_celsius']:.1f}°C (target: {b['target_min_c']:.0f}–{b['target_max_c']:.0f}°C)"
            )

    lines.append(f"\n── Expiry Risk (next {days_ahead} days) ──")
    if at_expiry.empty:
        lines.append(f"  No units expiring in next {days_ahead} days.")
    else:
        lines.append(
            f"  {len(at_expiry)} SKU-days with expiry risk | Est. Waste Value: ₹{total_waste:,.0f}"
        )
        for _, r in at_expiry.sort_values("expiry_date").head(10).iterrows():
            lines.append(
                f"  {r['sku_id']} – {r['name']}: {int(r.get('units_at_risk_of_expiry', 0))} units "
                f"| Expires: {r['expiry_date'].date()} | Shelf life: {int(r['shelf_life_days_remaining'])}d"
            )

    lines.append(f"\n── CRITICAL: Shelf Life < 3 Days ──")
    if critical_expiry.empty:
        lines.append("  No critically short shelf-life items.")
    else:
        lines.append(
            f"  *** URGENT: {len(critical_expiry)} records with < 3 days shelf life! ***"
        )
        for _, r in critical_expiry.iterrows():
            lines.append(
                f"  ALERT: {r['sku_id']} – {r['name']}: "
                f"{int(r['shelf_life_days_remaining'])} days remaining | "
                f"{int(r['units_in_cold_storage'])} units in storage"
            )

    lines.append(
        f"\n── Summary ──\n"
        f"  Total breach events (7d): {len(breaches)}\n"
        # BUG-036 fix: units_at_risk_of_expiry is 0 in 92% of rows (sparse synthetic field).
        # Use units_in_cold_storage for rows in the expiry window as the risk count.
        f"  Units at expiry risk: {int(at_expiry['units_in_cold_storage'].sum()) if not at_expiry.empty and 'units_in_cold_storage' in at_expiry.columns else int(at_expiry.get('units_at_risk_of_expiry', pd.Series([0])).sum())}\n"
        f"  Estimated waste value: ₹{total_waste:,.0f}\n"
        f"  CRITICAL items (< 3d): {len(critical_expiry)}"
    )
    return "\n".join(lines)


def tool_get_supplier_lead_time_tracker(supplier_name: str | None = None) -> str:
    """Track actual vs promised lead times per supplier, flag underperformers."""
    try:
        sp = get_supplier_perf()
    except FileNotFoundError as e:
        return f"Supplier performance data not available: {e}"

    # Get promised lead times from demand CSV
    try:
        dem = get_df()
        promised = (
            dem.groupby("supplier")["lead_time_days"]
            .mean()
            .reset_index()
            .rename(
                columns={"lead_time_days": "promised_lt", "supplier": "supplier_name"}
            )
        )
    except Exception:
        promised = pd.DataFrame(columns=["supplier_name", "promised_lt"])

    if supplier_name:
        sp = sp[sp["supplier_name"].str.contains(supplier_name, case=False, na=False)]
        if sp.empty:
            return f"Supplier '{supplier_name}' not found. Available: {', '.join(get_supplier_perf()['supplier_name'].unique())}"

    # Sort chronologically
    sp = sp.sort_values(["supplier_name", "review_month"])

    # Aggregate per supplier
    agg = (
        sp.groupby("supplier_name")
        .agg(
            avg_actual_lt=("lead_time_actual_days", "mean"),
            avg_otd_pct=("on_time_delivery_pct", "mean"),
            avg_fill_rate=("fill_rate_pct", "mean"),
            latest_otd=("on_time_delivery_pct", "last"),
            reviews=("review_month", "count"),
        )
        .reset_index()
    )
    agg = agg.merge(promised, on="supplier_name", how="left")
    agg["lt_variance_days"] = (agg["avg_actual_lt"] - agg["promised_lt"]).round(1)
    agg = agg.sort_values("avg_otd_pct")

    lines = [f"=== Supplier Lead Time Tracker ===\n"]

    if supplier_name and len(agg) == 1:
        # Detailed trend view
        r = agg.iloc[0]
        sup_data = sp[sp["supplier_name"] == r["supplier_name"]].tail(6)
        lines.append(f"── {r['supplier_name']} — Detailed Trend ──")
        lines.append(f"  Avg Actual Lead Time : {r['avg_actual_lt']:.1f} days")
        lines.append(
            f"  Promised Lead Time   : {r['promised_lt']:.1f} days"
            if pd.notna(r.get("promised_lt"))
            else "  Promised Lead Time   : N/A"
        )
        lines.append(
            f"  Lead Time Variance   : {r['lt_variance_days']:+.1f} days"
            if pd.notna(r.get("lt_variance_days"))
            else ""
        )
        lines.append(f"  Avg On-Time Delivery : {r['avg_otd_pct']:.1f}%")
        lines.append(f"  Latest OTD           : {r['latest_otd']:.1f}%")
        lines.append(f"  Avg Fill Rate        : {r['avg_fill_rate']:.1f}%")
        lines.append(f"\n  Last {len(sup_data)} months trend:")
        for _, row in sup_data.iterrows():
            flag = " ⚠" if row["on_time_delivery_pct"] < 90 else ""
            lines.append(
                f"    {str(row['review_month'])[:7]}: OTD={row['on_time_delivery_pct']:.1f}% | "
                f"LT={row['lead_time_actual_days']:.0f}d | Fill={row['fill_rate_pct']:.1f}%{flag}"
            )
    else:
        for _, r in agg.iterrows():
            flag = " *** UNDERPERFORMING ***" if r["avg_otd_pct"] < 90 else ""
            lines.append(
                f"  {r['supplier_name']}\n"
                f"    OTD: {r['avg_otd_pct']:.1f}% (latest: {r['latest_otd']:.1f}%) | "
                f"Actual LT: {r['avg_actual_lt']:.1f}d | "
                f"Fill Rate: {r['avg_fill_rate']:.1f}% | "
                f"Reviews: {int(r['reviews'])}{flag}\n"
            )

    underperforming = agg[agg["avg_otd_pct"] < 90]
    if not underperforming.empty:
        lines.append(
            f"\n*** {len(underperforming)} supplier(s) with OTD < 90%: "
            f"{', '.join(underperforming['supplier_name'].tolist())} ***"
        )
    return "\n".join(lines)


def tool_get_return_rate_analysis(
    category: str | None = None, brand: str | None = None
) -> str:
    """Analyse return rates by category and brand; flag SKUs with return_rate > 5%."""
    try:
        ret = get_returns()
        txn = get_transactions()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    if category:
        ret = ret[ret["category"].str.contains(category, case=False, na=False)]
        txn = txn[txn["category"].str.contains(category, case=False, na=False)]
    if brand:
        ret = ret[ret["brand"].str.contains(brand, case=False, na=False)]
        txn = txn[txn["brand"].str.contains(brand, case=False, na=False)]

    # Category-level
    cat_sold = (
        txn.groupby("category")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "sold"})
    )
    cat_ret = (
        ret.groupby("category")["quantity_returned"]
        .sum()
        .reset_index()
        .rename(columns={"quantity_returned": "returned"})
    )
    cat_rr = cat_sold.merge(cat_ret, on="category", how="left").fillna(0)
    cat_rr["return_rate_pct"] = (
        cat_rr["returned"] / cat_rr["sold"].replace(0, np.nan) * 100
    ).round(2)
    cat_rr = cat_rr.sort_values("return_rate_pct", ascending=False)

    # Brand-level
    br_sold = (
        txn.groupby("brand")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "sold"})
    )
    br_ret = (
        ret.groupby("brand")["quantity_returned"]
        .sum()
        .reset_index()
        .rename(columns={"quantity_returned": "returned"})
    )
    br_rr = br_sold.merge(br_ret, on="brand", how="left").fillna(0)
    br_rr["return_rate_pct"] = (
        br_rr["returned"] / br_rr["sold"].replace(0, np.nan) * 100
    ).round(2)
    br_rr = br_rr.sort_values("return_rate_pct", ascending=False)

    # Return reasons
    top_reasons = ret["return_reason"].value_counts().head(5)

    # SKU-level flags
    sku_sold = (
        txn.groupby("sku_id")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "sold"})
    )
    sku_ret = (
        ret.groupby("sku_id")["quantity_returned"]
        .sum()
        .reset_index()
        .rename(columns={"quantity_returned": "returned"})
    )
    sku_rr = sku_sold.merge(sku_ret, on="sku_id", how="left").fillna(0)
    sku_rr["return_rate_pct"] = (
        sku_rr["returned"] / sku_rr["sold"].replace(0, np.nan) * 100
    ).round(2)
    high_rr = sku_rr[sku_rr["return_rate_pct"] > 5].sort_values(
        "return_rate_pct", ascending=False
    )

    scope = []
    if category:
        scope.append(f"Category: {category}")
    if brand:
        scope.append(f"Brand: {brand}")
    scope_str = " | ".join(scope) if scope else "All"

    BENCHMARK = 3.0
    overall_ret = ret["quantity_returned"].sum()
    overall_sold = txn["quantity"].sum()
    overall_rate = round(overall_ret / max(overall_sold, 1) * 100, 2)

    lines = [f"=== Return Rate Analysis — {scope_str} ===\n"]
    lines.append(
        f"Overall Return Rate: {overall_rate:.2f}% "
        f"({'ABOVE' if overall_rate > BENCHMARK else 'BELOW'} industry benchmark of {BENCHMARK}%)\n"
    )

    lines.append("── By Category ──")
    for _, r in cat_rr.iterrows():
        flag = " *** HIGH ***" if r["return_rate_pct"] > 5 else ""
        lines.append(
            f"  {r['category']:30s} Sold: {int(r['sold']):>6,} | "
            f"Returned: {int(r['returned']):>5,} | Rate: {r['return_rate_pct']:.2f}%{flag}"
        )

    lines.append("\n── By Brand (Top 10) ──")
    for _, r in br_rr.head(10).iterrows():
        flag = " *** HIGH ***" if r["return_rate_pct"] > 5 else ""
        lines.append(
            f"  {r['brand']:25s} Sold: {int(r['sold']):>6,} | Rate: {r['return_rate_pct']:.2f}%{flag}"
        )

    lines.append("\n── Top Return Reasons ──")
    for reason, count in top_reasons.items():
        lines.append(f"  {reason}: {count:,} returns")

    if not high_rr.empty:
        lines.append(
            f"\n*** {len(high_rr)} SKUs with return rate > 5% (action needed): ***"
        )
        for _, r in high_rr.head(10).iterrows():
            lines.append(
                f"  {r['sku_id']}: {r['return_rate_pct']:.2f}% ({int(r['returned'])} / {int(r['sold'])})"
            )

    return "\n".join(lines)


def tool_get_dead_stock_analysis(days_no_movement: int = 60) -> str:
    """Find SKUs with no demand for N days; compute locked value and clearance pricing."""
    try:
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    latest_date = dem["date"].max()
    cutoff = latest_date - pd.Timedelta(days=days_no_movement)

    # Average daily demand in the dead-stock window per SKU
    window = dem[dem["date"] >= cutoff]
    avg_demand_window = window.groupby("sku_id")["demand"].mean()

    # Current inventory
    latest = dem[dem["date"] == latest_date][
        ["sku_id", "name", "category", "inventory", "cost_inr", "price_inr"]
    ].drop_duplicates("sku_id")

    dead = []
    for sku, avg_d in avg_demand_window.items():
        if avg_d < 0.5:  # near-zero demand
            row = latest[latest["sku_id"] == sku]
            if row.empty or int(row.iloc[0]["inventory"]) == 0:
                continue
            r = row.iloc[0]
            units = int(r["inventory"])
            cost = float(r["cost_inr"]) if pd.notna(r["cost_inr"]) else 0
            price = float(r["price_inr"]) if pd.notna(r["price_inr"]) else 0
            locked_value = units * cost
            holding_cost_month = locked_value * 0.02
            # To clear in 30 days with elasticity = -1.5
            # Need demand uplift = units/30. Current demand ≈ avg_d.
            # Discount needed: if avg_d ≈ 0, use 40% as base
            discount_pct = min(60, max(20, round(40 + (30 - avg_d * 30) * 0.5)))
            clearance_price = round(price * (1 - discount_pct / 100))
            dead.append(
                {
                    "sku_id": sku,
                    "name": r["name"],
                    "category": r["category"],
                    "units": units,
                    "locked_value_inr": locked_value,
                    "holding_cost_month_inr": holding_cost_month,
                    "suggested_discount_pct": discount_pct,
                    "clearance_price_inr": clearance_price,
                    "avg_daily_demand": round(avg_d, 2),
                }
            )

    if not dead:
        return f"No dead stock found (threshold: avg demand < 0.5/day over last {days_no_movement} days)."

    dead_df = pd.DataFrame(dead).sort_values("locked_value_inr", ascending=False)

    # BUG-10 fix: warn when cost_inr is 0 (missing from MySQL legacy schema)
    if dead_df.empty is False and dead_df["locked_value_inr"].sum() == 0:
        return (
            "⚠️ WARNING: cost_inr is 0 for all SKUs — locked value cannot be calculated.\n"
            "Dead stock SKUs exist but financial values are unavailable.\n"
            "Run: python db/migrate_huft.py to load full financial data.\n\n"
            f"Dead/slow-moving SKUs found (units only):\n"
            + "\n".join(
                f"  {r['sku_id']} — {r['name']} ({r['category']}): {int(r['units']):,} units | "
                f"Suggested clearance: {r['suggested_discount_pct']}% off"
                for _, r in dead_df.iterrows()
            )
        )
    total_locked = dead_df["locked_value_inr"].sum()
    total_holding = dead_df["holding_cost_month_inr"].sum()

    lines = [
        f"=== Dead Stock Analysis — Last {days_no_movement} Days ===\n",
        f"  Dead SKUs Found   : {len(dead_df)}",
        f"  Total Locked Value: ₹{total_locked:,.0f}",
        f"  Monthly Holding Cost: ₹{total_holding:,.0f} (2%/month)\n",
        "── SKU Details (ranked by locked value) ──",
    ]
    for _, r in dead_df.head(15).iterrows():
        lines.append(
            f"  {r['sku_id']} – {r['name']} ({r['category']})\n"
            f"    Units: {int(r['units']):,} | Locked: ₹{r['locked_value_inr']:,.0f} | "
            f"Holding/mo: ₹{r['holding_cost_month_inr']:,.0f}\n"
            f"    Avg Demand: {r['avg_daily_demand']:.2f}/day | "
            f"Recommended Clearance: {r['suggested_discount_pct']}% off → ₹{int(r['clearance_price_inr']):,}\n"
        )
    return "\n".join(lines)


def tool_get_competitive_price_analysis(
    sku_id: str | None = None, brand: str | None = None
) -> str:
    """Compare Pet Store prices with competitor prices via web search."""
    try:
        prod = get_products()
    except FileNotFoundError as e:
        return f"Product data not available: {e}"

    if sku_id:
        prod = prod[prod["sku_id"].str.upper() == sku_id.upper()]
    if brand:
        prod = prod[prod["brand"].str.contains(brand, case=False, na=False)]

    if prod.empty:
        return f"No products found for sku_id={sku_id}, brand={brand}."

    sample = prod.head(3)
    lines = [f"=== Competitive Price Analysis ===\n"]
    lines.append(
        "Note: Competitor prices fetched via web_search. "
        "Use the web_search tool with queries like "
        f"'\"Royal Canin Labrador\" price site:amazon.in OR site:flipkart.com' for live data.\n"
    )

    for _, r in sample.iterrows():
        query = f'"{r["name"]}" {r["brand"]} price site:amazon.in OR site:flipkart.com'
        huft_price = float(r["price_inr"]) if pd.notna(r.get("price_inr")) else 0
        lines.append(
            f"  {r['sku_id']} – {r['name']} ({r['brand']})\n"
            f"    Pet Store Price : ₹{huft_price:,.0f}\n"
            f"    Suggested web_search query: {query}\n"
        )

    lines.append(
        "── Recommendation ──\n"
        "Run web_search for each product to get live competitor prices.\n"
        "Compare Pet Store price with Amazon.in / Flipkart / PetSutra competitors.\n"
        "If Pet Store price is >10% above competitor: consider price match or bundle strategy.\n"
        "If Pet Store price is <5% above: highlight premium service / delivery advantage."
    )
    return "\n".join(lines)


def tool_get_new_product_launch_readiness(sku_id: str) -> str:
    """Assess new product launch health: inventory ramp, early stockouts, fill rate."""
    try:
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    sku_id = sku_id.upper().strip()
    sku_df = dem[dem["sku_id"] == sku_id].sort_values("date")
    if sku_df.empty:
        return f"SKU '{sku_id}' not found."

    name = sku_df["name"].iloc[0]
    dataset_start = dem["date"].min()
    launch_date = sku_df["date"].min()
    total_days = len(sku_df)

    # Only treat as "new" if the SKU started at least 7 days after the dataset began.
    # If it started on day 1 of the dataset, it was part of the original catalogue.
    days_after_start = (launch_date - dataset_start).days
    is_new_launch = days_after_start >= 7

    first_30 = sku_df.head(30)
    next_30 = sku_df.iloc[30:60]

    avg_d1 = float(first_30["demand"].mean())
    avg_d2 = float(next_30["demand"].mean()) if len(next_30) > 0 else 0
    demand_ramp = ((avg_d2 - avg_d1) / max(avg_d1, 1e-6) * 100) if avg_d2 > 0 else 0

    # Stockout days in first 30 days
    stockouts_30 = int((first_30["inventory"] == 0).sum())
    stockouts_all = int((sku_df["inventory"] == 0).sum())

    # Fill rate proxy: % of days with positive inventory
    fill_rate = round((sku_df["inventory"] > 0).sum() / len(sku_df) * 100, 1)

    # Current inventory
    latest = sku_df.iloc[-1]
    current_inv = int(latest["inventory"])
    current_demand = float(latest["demand"])
    dos = round(current_inv / max(current_demand, 1e-6), 1)

    # Health score (0-100)
    score = 100
    score -= min(30, stockouts_30 * 3)  # stockout penalty
    score -= min(20, (100 - fill_rate))  # fill rate penalty
    if demand_ramp < 0:
        score -= 15
    elif demand_ramp < 10:
        score -= 5
    health = (
        "EXCELLENT"
        if score >= 80
        else "GOOD"
        if score >= 60
        else "NEEDS ATTENTION"
        if score >= 40
        else "POOR"
    )

    lines = [
        f"=== New Product Launch Readiness: {sku_id} — {name} ===\n",
        f"  Launch Date    : {launch_date.date()}"
        + (
            ""
            if is_new_launch
            else " (note: SKU existed at dataset start — metrics cover full history)"
        ),
        f"  Days on Shelf  : {total_days}\n",
        f"── Demand Ramp ──",
        f"  First 30 days  : {avg_d1:.1f} units/day avg",
        f"  Next 30 days   : {avg_d2:.1f} units/day avg",
        f"  Demand Ramp    : {demand_ramp:+.1f}% ({'improving' if demand_ramp > 0 else 'declining'})\n",
        f"── Inventory Health ──",
        f"  Current Inventory  : {current_inv:,} units",
        f"  Days of Supply     : {dos:.1f} days",
        f"  Stockouts (first 30d): {stockouts_30} days",
        f"  Stockouts (all time) : {stockouts_all} days",
        f"  Overall Fill Rate  : {fill_rate}%\n",
        f"── Launch Health Score: {score}/100 — {health} ──",
        f"  Recommendation: {'Excellent launch trajectory. Monitor for reorder point.' if health == 'EXCELLENT' else 'Review supply strategy — early stockouts may suppress demand ramp.' if health in ('NEEDS ATTENTION', 'POOR') else 'Good progress. Ensure restock before hitting reorder point.'}",
    ]
    return "\n".join(lines)


def tool_get_customer_segmentation_insights(segment: str | None = None) -> str:
    """Customer segment analysis: AOV, frequency, LTV, preferred categories."""
    try:
        cust = get_customers()
        txn = get_transactions()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    if segment:
        cust = cust[cust["segment"].str.contains(segment, case=False, na=False)]
        txn = txn[txn["customer_segment"].str.contains(segment, case=False, na=False)]

    # Segment summary from customers CSV
    seg_agg = (
        cust.groupby("segment")
        .agg(
            customer_count=("customer_id", "count"),
            avg_ltv_inr=("lifetime_value_inr", "mean"),
            total_orders=("total_orders", "mean"),
            top_pet=(
                "pet_type",
                lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A",
            ),
            top_channel=(
                "channel_preference",
                lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A",
            ),
        )
        .reset_index()
        .sort_values("avg_ltv_inr", ascending=False)
    )

    # BUG-034 fix: AOV = total revenue / number of unique orders (txn_id),
    # not mean(revenue per line-item). A 3-SKU basket should count as 1 order.
    txn_agg = (
        txn.groupby("customer_segment")
        .agg(
            total_revenue=("net_revenue_inr", "sum"),
            n_orders=("txn_id", "nunique"),
            top_category=(
                "category",
                lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A",
            ),
        )
        .reset_index()
        .rename(columns={"customer_segment": "segment"})
    )
    txn_agg["avg_order_value"] = txn_agg["total_revenue"] / txn_agg["n_orders"].replace(
        0, np.nan
    )

    merged = seg_agg.merge(txn_agg, on="segment", how="left")

    lines = [
        f"=== Customer Segmentation Insights — {'All' if not segment else segment} ===\n"
    ]
    for _, r in merged.iterrows():
        aov = (
            f"₹{r['avg_order_value']:,.0f}"
            if pd.notna(r.get("avg_order_value"))
            else "N/A"
        )
        rev = (
            f"₹{r['total_revenue']:,.0f}" if pd.notna(r.get("total_revenue")) else "N/A"
        )
        lines.append(
            f"── {r['segment']} ──\n"
            f"  Customers        : {int(r['customer_count']):,}\n"
            f"  Avg LTV          : ₹{r['avg_ltv_inr']:,.0f}\n"
            f"  Avg Orders       : {r['total_orders']:.1f}/customer\n"
            f"  Avg Order Value  : {aov}\n"
            f"  Top Pet Type     : {r.get('top_pet', 'N/A')}\n"
            f"  Preferred Channel: {r.get('top_channel', 'N/A')}\n"
            f"  Top Category     : {r.get('top_category', 'N/A')}\n"
            f"  Total Revenue    : {rev}\n"
        )

    # Inventory allocation recommendation
    lines.append("── Inventory Allocation Recommendation ──")
    top_seg = merged.nlargest(1, "avg_ltv_inr")
    if not top_seg.empty:
        ts = top_seg.iloc[0]
        lines.append(
            f"  Prioritise {ts['segment']} segment (highest LTV: ₹{ts['avg_ltv_inr']:,.0f})\n"
            f"  Ensure availability of '{ts.get('top_category', 'N/A')}' products for this segment\n"
            f"  Channel focus: {ts['top_channel']}"
        )
    return "\n".join(lines)


def tool_generate_purchase_order(
    urgency: str = "all", supplier_name: str | None = None
) -> str:
    """Generate complete purchase orders for all SKUs below reorder point."""
    try:
        dem = get_df()
        prod = get_products()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    latest_date = dem["date"].max()
    latest = dem[dem["date"] == latest_date].copy()
    cutoff30 = latest_date - pd.Timedelta(days=30)
    cutoff90 = latest_date - pd.Timedelta(days=90)

    avg30 = dem[dem["date"] >= cutoff30].groupby("sku_id")["demand"].mean()
    avg90 = dem[dem["date"] >= cutoff90].groupby("sku_id")["demand"].mean()

    latest["avg30"] = latest["sku_id"].map(avg30).fillna(0)
    latest["avg90"] = latest["sku_id"].map(avg90).fillna(0)

    # BUG-002 fix: correct safety-stock formula  SS = Z × σ × √(LT)  (Z=1.65 → 95%)
    # BUG-003 fix: order_qty covers forecast-period demand + safety_stock - usable_inv
    std90 = dem[dem["date"] >= cutoff90].groupby("sku_id")["demand"].std().fillna(0)
    latest["sigma"] = latest["sku_id"].map(std90).fillna(latest["avg30"] * 0.25)
    latest["safety_stock"] = (
        1.65 * latest["sigma"] * np.sqrt(latest["lead_time_days"])
    ).round(0)
    latest["reorder_point"] = (
        latest["avg30"] * latest["lead_time_days"] + latest["safety_stock"]
    ).round(0)
    latest["usable_inv"] = (latest["inventory"] - latest["safety_stock"]).clip(lower=0)
    # Order enough to cover the lead-time demand plus restore safety stock buffer
    latest["order_qty"] = (
        (
            latest["avg90"] * latest["lead_time_days"]
            + latest["safety_stock"]
            - latest["inventory"]
        )
        .clip(lower=0)
        .round(0)
        .astype(int)
    )
    latest["est_cost_inr"] = (latest["order_qty"] * latest["cost_inr"].fillna(0)).round(
        0
    )
    latest["needs_reorder"] = latest["inventory"] <= latest["reorder_point"]

    if supplier_name:
        latest = latest[
            latest["supplier"].str.contains(supplier_name, case=False, na=False)
        ]

    needs = latest[latest["needs_reorder"] & (latest["order_qty"] > 0)].copy()
    if needs.empty:
        return f"No reorder needed{' for supplier: ' + supplier_name if supplier_name else ''}."

    needs["dos"] = (
        (needs["inventory"] / needs["avg30"].replace(0, np.nan)).fillna(999).round(1)
    )
    needs["urgency_flag"] = needs["dos"].apply(
        lambda d: "CRITICAL" if d < 7 else "WARNING" if d < 14 else "NORMAL"
    )

    if urgency.lower() == "critical":
        needs = needs[needs["urgency_flag"] == "CRITICAL"]
    elif urgency.lower() == "warning":
        needs = needs[needs["urgency_flag"].isin(["CRITICAL", "WARNING"])]

    needs = needs.sort_values("dos")

    total_value = needs["est_cost_inr"].sum()
    lines = [
        f"╔══ PURCHASE ORDER — {latest_date.date()} ══╗\n",
        f"  SKUs to Order   : {len(needs)}",
        f"  Total PO Value  : ₹{total_value:,.0f}",
        f"  Filter          : Urgency={urgency.upper()} | Supplier={supplier_name or 'All'}\n",
        "── Line Items (sorted by urgency) ──",
    ]

    by_supplier = needs.groupby("supplier")
    for sup, grp in by_supplier:
        sup_total = grp["est_cost_inr"].sum()
        lines.append(f"\n  Supplier: {sup} | Sub-total: ₹{sup_total:,.0f}")
        for _, r in grp.iterrows():
            lines.append(
                f"  [{r['urgency_flag']}] {r['sku_id']} – {r['name']}\n"
                f"      Current Stock: {int(r['inventory']):,} | DoS: {r['dos']:.1f}d | "
                f"Lead Time: {int(r['lead_time_days'])}d\n"
                f"      Order Qty: {int(r['order_qty']):,} units | "
                f"Est. Cost: ₹{int(r['est_cost_inr']):,} | "
                f"Expected Delivery: ~{int(r['lead_time_days'])} days\n"
            )
    return "\n".join(lines)


def tool_get_promotion_inventory_impact(
    promo_id: str | None = None,
    days_before: int = 7,
    days_after: int = 14,
) -> str:
    """Analyse demand lift, stockouts, and restock lag for promotions."""
    try:
        promos = get_promotions()
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    if promo_id:
        promos = promos[promos["promo_id"].str.contains(promo_id, case=False, na=False)]
        if promos.empty:
            return f"Promo '{promo_id}' not found."

    lines = [f"=== Promotion Inventory Impact Analysis ===\n"]

    for _, promo in promos.iterrows():
        start = pd.Timestamp(promo["start_date"])
        end = pd.Timestamp(promo["end_date"])
        pre_start = start - pd.Timedelta(days=days_before)
        post_end = end + pd.Timedelta(days=days_after)
        cat = promo.get("target_category", "")

        cat_dem = (
            dem[dem["category"].str.contains(cat, case=False, na=False)] if cat else dem
        )

        pre = cat_dem[(cat_dem["date"] >= pre_start) & (cat_dem["date"] < start)]
        during = cat_dem[(cat_dem["date"] >= start) & (cat_dem["date"] <= end)]
        post = cat_dem[(cat_dem["date"] > end) & (cat_dem["date"] <= post_end)]

        avg_pre = float(pre["demand"].mean()) if len(pre) > 0 else 0
        avg_during = float(during["demand"].mean()) if len(during) > 0 else 0
        avg_post = float(post["demand"].mean()) if len(post) > 0 else 0
        lift_pct = ((avg_during - avg_pre) / max(avg_pre, 1e-6)) * 100

        stockout_skus = during[during["inventory"] == 0]["sku_id"].nunique()

        lines.append(
            f"── {promo['name']} ({promo['promo_id']}) ──\n"
            f"  Category : {cat} | Discount: {promo.get('discount_pct', 'N/A')}% | "
            f"Channel: {promo.get('channel', 'N/A')}\n"
            f"  Period   : {start.date()} → {end.date()} "
            # BUG-056 fix: duration_days may be NaN even when the key exists;
            # fall back to computed duration when it is missing or NaN.
            f"({int(promo['duration_days']) if pd.notna(promo.get('duration_days')) else (end - start).days} days)\n"
            f"  Demand (pre/during/post): {avg_pre:.1f} / {avg_during:.1f} / {avg_post:.1f} units/day\n"
            f"  Demand Lift During Promo: {lift_pct:+.1f}%\n"
            f"  SKUs with Stockout During Promo: {stockout_skus}\n"
            f"  Revenue Generated: ₹{float(promo.get('revenue_generated_inr', 0)):,.0f}\n"
            f"  Restock Lag: Post-promo demand {avg_post:.1f}/day vs baseline {avg_pre:.1f}/day\n"
        )

    return "\n".join(lines)


def tool_get_channel_revenue_attribution(
    channel: str | None = None, period: str = "last_90_days"
) -> str:
    """Break down revenue, units, margin, and top SKUs by channel."""
    try:
        txn = get_transactions()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    if period == "last_90_days":
        cutoff = txn["date"].max() - pd.Timedelta(days=90)
        txn = txn[txn["date"] >= cutoff]
    elif period == "last_30_days":
        cutoff = txn["date"].max() - pd.Timedelta(days=30)
        txn = txn[txn["date"] >= cutoff]

    if channel:
        txn = txn[txn["channel"].str.contains(channel, case=False, na=False)]

    ch_agg = (
        txn.groupby("channel")
        .agg(
            total_revenue_inr=("net_revenue_inr", "sum"),
            total_units=("quantity", "sum"),
            total_margin_inr=("gross_margin_inr", "sum"),
            transactions=("txn_id", "count"),
        )
        .reset_index()
    )
    ch_agg["margin_pct"] = (
        ch_agg["total_margin_inr"]
        / ch_agg["total_revenue_inr"].replace(0, np.nan)
        * 100
    ).round(1)
    ch_agg["avg_order_value"] = (
        ch_agg["total_revenue_inr"] / ch_agg["transactions"].replace(0, np.nan)
    ).round(0)
    total_rev = ch_agg["total_revenue_inr"].sum()
    ch_agg["revenue_share_pct"] = (ch_agg["total_revenue_inr"] / total_rev * 100).round(
        1
    )
    ch_agg = ch_agg.sort_values("total_revenue_inr", ascending=False)

    lines = [f"=== Channel Revenue Attribution — {period} ===\n"]
    lines.append(f"  Total Revenue: ₹{total_rev:,.0f}\n")

    for _, r in ch_agg.iterrows():
        lines.append(
            f"── {r['channel']} ({r['revenue_share_pct']:.1f}% of revenue) ──\n"
            f"  Revenue : ₹{r['total_revenue_inr']:,.0f}\n"
            f"  Units   : {int(r['total_units']):,}\n"
            f"  Margin  : {r['margin_pct']:.1f}%\n"
            f"  Avg Order Value: ₹{r['avg_order_value']:,.0f}\n"
            f"  Transactions: {int(r['transactions']):,}\n"
        )
        # Top SKUs per channel
        ch_skus = (
            txn[txn["channel"] == r["channel"]]
            .groupby("sku_id")["net_revenue_inr"]
            .sum()
            .nlargest(3)
        )
        if not ch_skus.empty:
            lines.append(
                f"  Top SKUs: "
                + " | ".join(f"{sku} (₹{rev:,.0f})" for sku, rev in ch_skus.items())
                + "\n"
            )

    return "\n".join(lines)


def tool_get_markdown_optimization(category: str | None = None) -> str:
    """Find overstocked SKUs and generate clearance discount recommendations."""
    try:
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    latest_date = dem["date"].max()
    latest = dem[dem["date"] == latest_date].copy()
    cutoff90 = latest_date - pd.Timedelta(days=90)
    avg90 = dem[dem["date"] >= cutoff90].groupby("sku_id")["demand"].mean()

    latest["avg90_demand"] = latest["sku_id"].map(avg90).fillna(0)
    latest["days_to_clear"] = (
        (latest["inventory"] / latest["avg90_demand"].replace(0, np.nan))
        .fillna(9999)
        .round(0)
    )
    # BUG-014 fix: overstock = > 3 months forward demand (90 days), not 270 days
    latest["overstock_flag"] = latest["inventory"] > 3 * (latest["avg90_demand"] * 30)

    if category:
        latest = latest[latest["category"].str.contains(category, case=False, na=False)]

    overstocked = latest[latest["overstock_flag"]].copy()
    if overstocked.empty:
        return f"No overstocked SKUs found{' in category: ' + category if category else ''}."

    # Calculate suggested discount to clear in 30 days
    # Based on elasticity estimate: need avg daily demand = inventory/30
    overstocked["required_daily_demand"] = overstocked["inventory"] / 30
    overstocked["demand_uplift_needed"] = (
        overstocked["required_daily_demand"]
        / overstocked["avg90_demand"].replace(0, np.nan)
        - 1
    ).fillna(0) * 100
    # Elasticity -1.5: discount = uplift_needed / 1.5
    overstocked["suggested_discount_pct"] = (
        (overstocked["demand_uplift_needed"] / 1.5).clip(5, 60).round(0)
    )
    overstocked["revenue_at_discount"] = (
        overstocked["inventory"]
        * overstocked["price_inr"].fillna(0)
        * (1 - overstocked["suggested_discount_pct"] / 100)
    )
    overstocked["holding_cost_30d"] = (
        overstocked["inventory"] * overstocked["cost_inr"].fillna(0) * 0.02
    )
    overstocked["savings_vs_holding"] = (
        overstocked["revenue_at_discount"] - overstocked["holding_cost_30d"]
    )

    overstocked = overstocked.sort_values("savings_vs_holding", ascending=False)
    total_overstock_value = (
        overstocked["inventory"] * overstocked["cost_inr"].fillna(0)
    ).sum()

    lines = [
        f"=== Markdown Optimization — {'All Categories' if not category else category} ===\n",
        f"  Overstocked SKUs  : {len(overstocked)}",
        f"  Total Overstock Value: ₹{total_overstock_value:,.0f}\n",
        "── Clearance Recommendations (ranked by savings potential) ──",
    ]
    for _, r in overstocked.head(12).iterrows():
        lines.append(
            f"  {r['sku_id']} – {r['name']} ({r['category']})\n"
            f"    Inventory: {int(r['inventory']):,} units | 90d avg demand: {r['avg90_demand']:.1f}/day\n"
            f"    Days to clear at current rate: {int(r['days_to_clear']):,}d\n"
            f"    Suggested Discount: {int(r['suggested_discount_pct'])}% | "
            f"Revenue at discount: ₹{r['revenue_at_discount']:,.0f}\n"
            f"    Holding cost (30d): ₹{r['holding_cost_30d']:,.0f} | "
            f"Savings vs holding: ₹{r['savings_vs_holding']:,.0f}\n"
        )
    return "\n".join(lines)


def tool_get_marketing_campaign_recommendations(months_ahead: int = 3) -> str:
    """Identify top 5 categories to promote and top 5 to avoid."""
    try:
        dem = get_df()
        promos = get_promotions()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    latest_date = dem["date"].max()
    latest = dem[dem["date"] == latest_date].copy()
    cutoff90 = latest_date - pd.Timedelta(days=90)
    avg90 = (
        dem[dem["date"] >= cutoff90]
        .groupby(["sku_id", "category"])[["demand", "inventory", "margin_pct"]]
        .mean()
        .reset_index()
    )

    cat_stats = (
        avg90.groupby("category")
        .agg(
            avg_inventory=("inventory", "mean"),
            avg_demand=("demand", "mean"),
            avg_margin=("margin_pct", "mean"),
        )
        .reset_index()
    )
    cat_stats["days_of_supply"] = (
        cat_stats["avg_inventory"] / cat_stats["avg_demand"].replace(0, np.nan)
    ).fillna(0)
    cat_stats["is_overstocked"] = cat_stats["days_of_supply"] > 60
    cat_stats["is_understocked"] = cat_stats["days_of_supply"] < 14

    # BUG-018 fix: look ahead 3 months (not 2) for seasonal planning
    # BUG-20 fix: use months_ahead parameter instead of hardcoded range(3)
    current_month = pd.Timestamp.today().month
    upcoming_months = [(current_month + i - 1) % 12 + 1 for i in range(months_ahead)]
    dem_copy = dem.copy()
    dem_copy["month"] = dem_copy["date"].dt.month
    seasonal_hot = (
        dem_copy[dem_copy["month"].isin(upcoming_months)]
        .groupby("category")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "seasonal_demand"})
    )
    annual_avg = (
        dem_copy.groupby("category")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "annual_avg"})
    )
    seasonal = seasonal_hot.merge(annual_avg, on="category")
    seasonal["seasonal_index"] = (
        seasonal["seasonal_demand"] / seasonal["annual_avg"].replace(0, np.nan) * 100
    ).round(1)

    cat_stats = cat_stats.merge(
        seasonal[["category", "seasonal_index"]], on="category", how="left"
    )

    # PROMOTE: overstocked OR high margin + not understocked
    promote_candidates = cat_stats[
        (
            cat_stats["is_overstocked"]
            | (cat_stats["avg_margin"] > cat_stats["avg_margin"].quantile(0.6))
        )
        & (~cat_stats["is_understocked"])
    ].sort_values("avg_margin", ascending=False)

    # AVOID: understocked
    avoid_candidates = cat_stats[cat_stats["is_understocked"]].sort_values(
        "days_of_supply"
    )

    lines = ["=== Marketing Campaign Recommendations ===\n"]
    lines.append("── TOP 5 CATEGORIES TO PROMOTE NOW ──")
    for i, (_, r) in enumerate(promote_candidates.head(5).iterrows(), 1):
        reason = (
            "High margin"
            if r["avg_margin"] > cat_stats["avg_margin"].median()
            else "Overstocked — needs demand stimulus"
        )
        lines.append(
            f"  {i}. {r['category']}\n"
            f"     Days of Supply: {r['days_of_supply']:.0f}d | Margin: {r['avg_margin']:.1f}% | "
            f"Seasonal Index: {r.get('seasonal_index', 100):.0f}\n"
            f"     Reason: {reason}\n"
        )

    lines.append("── TOP 5 CATEGORIES TO AVOID PROMOTING (UNDERSTOCKED) ──")
    for i, (_, r) in enumerate(avoid_candidates.head(5).iterrows(), 1):
        lines.append(
            f"  {i}. {r['category']} — AVOID: Only {r['days_of_supply']:.0f}d supply. "
            f"Promoting would cause stockouts.\n"
        )

    if avoid_candidates.empty:
        lines.append(
            "  All categories adequately stocked — no category is off-limits for promotion."
        )

    return "\n".join(lines)


def tool_get_inventory_financial_summary(period: str = "current") -> str:
    """CFO-level inventory financial summary."""
    try:
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    latest_date = dem["date"].max()
    latest = dem[dem["date"] == latest_date].copy()

    # Ensure numeric columns
    for col in ["inventory", "cost_inr", "price_inr", "margin_pct", "demand"]:
        if col in latest.columns:
            latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0)

    total_inventory_value = (latest["inventory"] * latest["cost_inr"]).sum()
    total_retail_value = (latest["inventory"] * latest["price_inr"]).sum()
    potential_gross_margin = total_retail_value - total_inventory_value

    # BUG-011: warn when cost/margin data is missing (defaults to 0 from _normalise_demand_df)
    _data_quality_note = ""
    if total_inventory_value == 0 and latest["inventory"].sum() > 0:
        _data_quality_note = (
            "\n⚠️  WARNING: cost_inr is 0 for all SKUs — financial values are understated.\n"
            "   Run: python db/migrate_huft.py  to load full financial data.\n"
        )

    # Dead stock (no movement 60 days)
    cutoff60 = latest_date - pd.Timedelta(days=60)
    window = dem[dem["date"] >= cutoff60]
    avg60 = window.groupby("sku_id")["demand"].mean()
    dead_skus = avg60[avg60 < 0.5].index
    dead_rows = latest[latest["sku_id"].isin(dead_skus)]
    dead_stock_value = (dead_rows["inventory"] * dead_rows["cost_inr"]).sum()

    # Stockout lost revenue
    cutoff30 = latest_date - pd.Timedelta(days=30)
    recent = dem[dem["date"] >= cutoff30].copy()
    for col in ["demand", "inventory", "price_inr"]:
        if col in recent.columns:
            recent[col] = pd.to_numeric(recent[col], errors="coerce").fillna(0)
    stockout_days = recent[(recent["inventory"] == 0) & (recent["demand"] > 0)]
    stockout_lost = (stockout_days["demand"] * stockout_days["price_inr"]).sum()

    # BUG-006 fix: working capital days = avg inventory value / avg daily revenue
    # Use average daily inventory value (not snapshot) divided by average daily revenue
    dem_recent = dem[dem["date"] >= cutoff30]
    avg_daily_inv_value = (
        dem_recent.groupby("date")
        .apply(lambda g: (g["inventory"] * g["cost_inr"]).sum())
        .mean()
    )
    avg_daily_sales = (
        dem_recent.groupby("date")
        .apply(lambda g: (g["demand"] * g["price_inr"]).sum())
        .mean()
    )
    working_capital_days = (
        (avg_daily_inv_value / max(avg_daily_sales, 1)) if avg_daily_sales > 0 else 0
    )

    # Category breakdown — using assign + groupby agg (avoids deprecated apply→Series)
    latest_fin = latest.assign(
        inv_value=latest["inventory"] * latest["cost_inr"],
        ret_value=latest["inventory"] * latest["price_inr"],
    )
    cat_breakdown = (
        latest_fin.groupby("category")
        .agg(inventory_value=("inv_value", "sum"), retail_value=("ret_value", "sum"))
        .reset_index()
    )
    cat_breakdown["margin_inr"] = (
        cat_breakdown["retail_value"] - cat_breakdown["inventory_value"]
    )
    cat_breakdown = cat_breakdown.sort_values("inventory_value", ascending=False)

    lines = [
        f"╔══ INVENTORY FINANCIAL SUMMARY ({latest_date.date()}) ══╗\n",
        *([_data_quality_note] if _data_quality_note else []),
        f"  Total Inventory Value (cost)  : ₹{total_inventory_value:,.0f}",
        f"  Total Retail Value            : ₹{total_retail_value:,.0f}",
        f"  Potential Gross Margin        : ₹{potential_gross_margin:,.0f} ({potential_gross_margin / max(total_retail_value, 1) * 100:.1f}%)",
        f"  Dead Stock Value              : ₹{dead_stock_value:,.0f}",
        f"  Stockout Lost Revenue (30d)   : ₹{stockout_lost:,.0f}",
        f"  Working Capital Days          : {working_capital_days:.0f} days\n",
        "── By Category ──",
    ]
    for _, r in cat_breakdown.iterrows():
        lines.append(
            f"  {r['category']:30s} "
            f"Inv Value: ₹{r['inventory_value']:>12,.0f} | "
            f"Retail: ₹{r['retail_value']:>12,.0f} | "
            f"Margin: ₹{r['margin_inr']:>10,.0f}"
        )
    lines.append(f"\n╚══ End of Financial Summary ══╝")
    return "\n".join(lines)


def tool_get_customer_cohort_demand_analysis(cohort_months: int = 3) -> str:
    """Quarterly customer cohort analysis: LTV, retention, top products per cohort."""
    try:
        cust = get_customers()
        txn = get_transactions()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    # Build cohorts based on join_date
    cust = cust.copy()
    cust["join_period"] = (
        cust["joined_date"].dt.to_period(f"{cohort_months}M").astype(str)
    )

    # Merge customers with transactions by segment proxy (no direct customer_id in txn)
    # Use segment-level analysis as best proxy
    cohort_summary = (
        cust.groupby("join_period")
        .agg(
            customers=("customer_id", "count"),
            avg_ltv_inr=("lifetime_value_inr", "mean"),
            avg_orders=("total_orders", "mean"),
            top_pet=(
                "pet_type",
                lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A",
            ),
            top_channel=(
                "channel_preference",
                lambda x: x.value_counts().index[0] if len(x) > 0 else "N/A",
            ),
        )
        .reset_index()
        .sort_values("join_period")
    )

    lines = [
        f"=== Customer Cohort Demand Analysis (Cohort Size: {cohort_months}M) ===\n"
    ]
    for _, r in cohort_summary.iterrows():
        lines.append(
            f"── Cohort: {r['join_period']} ──\n"
            f"  Customers       : {int(r['customers']):,}\n"
            f"  Avg LTV         : ₹{r['avg_ltv_inr']:,.0f}\n"
            f"  Avg Orders      : {r['avg_orders']:.1f}\n"
            f"  Top Pet Type    : {r['top_pet']}\n"
            f"  Top Channel     : {r['top_channel']}\n"
        )

    # Retention insight
    if len(cohort_summary) > 1:
        oldest = cohort_summary.iloc[0]
        newest = cohort_summary.iloc[-1]
        ltv_growth = (
            (newest["avg_ltv_inr"] - oldest["avg_ltv_inr"])
            / max(oldest["avg_ltv_inr"], 1)
        ) * 100
        lines.append(
            f"── Cohort Trend ──\n"
            f"  LTV growth from {oldest['join_period']} to {newest['join_period']}: "
            f"{ltv_growth:+.1f}%\n"
            f"  {'LTV improving — newer cohorts spending more' if ltv_growth > 0 else 'LTV declining — newer customers spending less; review acquisition strategy'}"
        )
    return "\n".join(lines)


def tool_get_store_level_demand_intelligence(
    store_id: str | None = None, city: str | None = None
) -> str:
    """Store-level demand patterns: uniqueness vs national average, stockouts, top SKUs."""
    try:
        txn = get_transactions()
        stores = get_stores()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    if store_id:
        txn = txn[
            txn["store_id"].astype(str).str.contains(store_id, case=False, na=False)
        ]
        stores = stores[
            stores["store_id"].astype(str).str.contains(store_id, case=False, na=False)
        ]
    if city:
        txn = txn[txn["city"].str.contains(city, case=False, na=False)]
        stores = stores[stores["city"].str.contains(city, case=False, na=False)]

    # BUG-028 fix: national average = avg daily total units sold per category,
    # not avg transaction line-item quantity (which is always ~1.5-2).
    date_col = "date" if "date" in txn.columns else "txn_date"
    national_cat = (
        txn.groupby(["category", date_col])["quantity"]
        .sum()  # total units sold per day per category
        .groupby(level="category")
        .mean()  # avg daily units
        .reset_index()
        .rename(columns={"quantity": "national_avg"})
    )

    # City-level breakdown: also use daily totals, not per-transaction means
    city_stats = (
        txn.groupby(["city", "category", date_col])["quantity"]
        .sum()
        .reset_index()
        .groupby(["city", "category"])
        .agg(avg_qty=("quantity", "mean"))
        .reset_index()
    )
    rev_stats = (
        txn.groupby(["city", "category"])["net_revenue_inr"]
        .sum()
        .reset_index()
        .rename(columns={"net_revenue_inr": "total_revenue"})
    )
    city_stats = city_stats.merge(rev_stats, on=["city", "category"], how="left")
    city_stats = city_stats.merge(national_cat, on="category", how="left")
    city_stats["vs_national_pct"] = (
        (city_stats["avg_qty"] - city_stats["national_avg"])
        / city_stats["national_avg"].replace(0, np.nan)
        * 100
    ).round(1)

    # Top cities by revenue
    city_rev = txn.groupby("city")["net_revenue_inr"].sum().nlargest(10)

    # Store overview
    store_info = stores.head(10) if not stores.empty else pd.DataFrame()

    lines = [f"=== Store-Level Demand Intelligence ===\n"]

    if not store_info.empty:
        lines.append("── Store Network ──")
        for _, s in store_info.iterrows():
            lines.append(
                f"  {s['store_id']} | {s['city']}, {s['state']} | "
                f"{s['store_type']} | {s['size_sqft']:,} sqft | Opened: {s['opened_year']}"
            )
        lines.append("")

    lines.append("── Top Cities by Revenue ──")
    for city_name, rev in city_rev.items():
        lines.append(f"  {city_name}: ₹{rev:,.0f}")

    lines.append("\n── City vs National Average Demand (top deviations) ──")
    extreme = (
        city_stats[city_stats["vs_national_pct"].abs() > 20]
        .sort_values("vs_national_pct", ascending=False)
        .head(15)
    )
    for _, r in extreme.iterrows():
        sign = "above" if r["vs_national_pct"] > 0 else "below"
        lines.append(
            f"  {r['city']:20s} | {r['category']:25s} | "
            f"{abs(r['vs_national_pct']):.0f}% {sign} national avg"
        )

    # Rebalancing opportunities
    lines.append("\n── Rebalancing Opportunities ──")
    surplus_cities = city_stats[city_stats["vs_national_pct"] < -25]
    if not surplus_cities.empty:
        for _, r in surplus_cities.head(5).iterrows():
            lines.append(
                f"  {r['city']} has LOW demand for {r['category']} "
                f"(-{abs(r['vs_national_pct']):.0f}% vs national). "
                f"Consider inventory rebalancing or targeted promotion."
            )
    else:
        lines.append("  No significant rebalancing needed based on current data.")

    return "\n".join(lines)


def tool_get_supplier_negotiation_brief(supplier_name: str | None = None) -> str:
    """Generate negotiation brief: leverage score, talking points, YoY volume."""
    try:
        sp = get_supplier_perf()
        dem = get_df()
    except FileNotFoundError as e:
        return f"Data not available: {e}"

    if supplier_name:
        sp_filtered = sp[
            sp["supplier_name"].str.contains(supplier_name, case=False, na=False)
        ]
        dem_filtered = dem[
            dem["supplier"].str.contains(supplier_name, case=False, na=False)
        ]
    else:
        sp_filtered = sp
        dem_filtered = dem

    suppliers = sp_filtered["supplier_name"].unique()
    lines = [f"=== Supplier Negotiation Brief ===\n"]

    for sup in suppliers:
        sup_sp = sp_filtered[sp_filtered["supplier_name"] == sup].sort_values(
            "review_month"
        )
        sup_dem = dem[dem["supplier"] == sup]

        avg_otd = float(sup_sp["on_time_delivery_pct"].mean()) if len(sup_sp) > 0 else 0
        latest_otd = (
            float(sup_sp["on_time_delivery_pct"].iloc[-1]) if len(sup_sp) > 0 else 0
        )
        avg_fill = float(sup_sp["fill_rate_pct"].mean()) if len(sup_sp) > 0 else 0
        avg_defect = float(sup_sp["defect_rate_pct"].mean()) if len(sup_sp) > 0 else 0
        actual_lt = (
            float(sup_sp["lead_time_actual_days"].mean()) if len(sup_sp) > 0 else 0
        )
        promised_lt = (
            float(sup_dem["lead_time_days"].mean()) if len(sup_dem) > 0 else actual_lt
        )
        num_skus = int(sup_dem["sku_id"].nunique()) if len(sup_dem) > 0 else 0

        # YoY volume
        sup_dem_copy = sup_dem.copy()
        sup_dem_copy["year"] = sup_dem_copy["date"].dt.year
        yoy = sup_dem_copy.groupby("year")["demand"].sum()
        yoy_growth = 0.0
        if len(yoy) >= 2:
            years = sorted(yoy.index)
            yoy_growth = (
                (yoy[years[-1]] - yoy[years[-2]]) / max(yoy[years[-2]], 1)
            ) * 100

        # Stockout incidents caused
        stockout_incidents = int(
            ((sup_dem["inventory"] == 0) & (sup_dem["demand"] > 0)).sum()
        )

        # Leverage score (0-10): higher = more leverage for buyer
        score = 5.0
        if avg_otd < 90:
            score += 2.0
        if avg_otd < 80:
            score += 1.0
        if avg_defect > 1.0:
            score += 1.0
        if actual_lt > promised_lt + 3:
            score += 1.0
        if stockout_incidents > 20:
            score += 1.0
        # BUG-9 fix: declining volume → supplier needs our business more → MORE buyer leverage
        if yoy_growth < 0:
            score += (
                1.0  # supplier needs us more when we're buying less → higher leverage
            )
        leverage_score = min(10, max(0, score))

        # Talking points
        talking_points = []
        if avg_otd < 90:
            talking_points.append(
                f"OTD is {avg_otd:.1f}% — below 90% SLA. Request penalty clause or SLA rebate."
            )
        if actual_lt > promised_lt + 2:
            talking_points.append(
                f"Actual LT ({actual_lt:.1f}d) exceeds contracted LT ({promised_lt:.1f}d). Negotiate LT reduction or express shipment coverage."
            )
        if avg_defect > 0.5:
            talking_points.append(
                f"Defect rate {avg_defect:.2f}% — above target. Request quality guarantee or credit for defects."
            )
        if stockout_incidents > 10:
            talking_points.append(
                f"{stockout_incidents} stockout incidents linked to this supplier. Negotiate VMI or consignment stock arrangement."
            )
        if yoy_growth > 20:
            talking_points.append(
                f"Volume grew {yoy_growth:.0f}% YoY — use as leverage for better pricing or priority allocation."
            )
        if not talking_points:
            talking_points.append(
                "Supplier performing well — focus negotiation on price reduction or extended payment terms."
            )

        lines.append(
            f"── {sup} ──\n"
            f"  SKUs Supplied    : {num_skus}\n"
            f"  Avg OTD          : {avg_otd:.1f}% (latest: {latest_otd:.1f}%)\n"
            f"  Avg Fill Rate    : {avg_fill:.1f}%\n"
            f"  Avg Defect Rate  : {avg_defect:.2f}%\n"
            f"  Actual vs Contract LT: {actual_lt:.1f}d vs {promised_lt:.1f}d\n"
            f"  Stockout Incidents: {stockout_incidents}\n"
            f"  YoY Volume Growth : {yoy_growth:+.1f}%\n"
            f"  Buyer Leverage Score: {leverage_score:.1f}/10\n"
            f"\n  Talking Points:"
        )
        for pt in talking_points:
            lines.append(f"    • {pt}")
        lines.append("")

    return "\n".join(lines)


def tool_get_product_recommendation(
    pet_type: str,
    breed: str | None = None,
    age_months: int | None = None,
    health_concern: str | None = None,
) -> str:
    """Recommend Pet Store products based on pet_type, breed, age, and health concern."""
    try:
        prod = get_products()
        dem = get_df()
    except FileNotFoundError as e:
        return f"Product data not available: {e}"

    # Filter by pet_type
    filtered = prod[
        prod["pet_type"].str.contains(pet_type, case=False, na=False)
    ].copy()
    if filtered.empty:
        available = prod["pet_type"].unique().tolist()
        return f"No products found for pet_type='{pet_type}'. Available: {available}"

    # Filter by breed suitability
    if breed and "breed_suitability" in filtered.columns:
        breed_filtered = filtered[
            filtered["breed_suitability"].str.contains(breed, case=False, na=False)
            | filtered["breed_suitability"].str.contains(
                "All Breeds", case=False, na=False
            )
        ]
        if not breed_filtered.empty:
            filtered = breed_filtered

    # Filter by life stage based on age_months
    if age_months is not None and "life_stage" in filtered.columns:
        if age_months <= 12:
            life_stage = "Puppy" if pet_type.lower() in ("dog", "dogs") else "Kitten"
        elif age_months <= 84:
            life_stage = "Adult"
        else:
            life_stage = "Senior"

        stage_filtered = filtered[
            filtered["life_stage"].str.contains(life_stage, case=False, na=False)
            | filtered["life_stage"].str.contains("All", case=False, na=False)
        ]
        if not stage_filtered.empty:
            filtered = stage_filtered
    else:
        life_stage = "All"

    # Filter by health concern
    if health_concern:
        concern_filtered = filtered[
            filtered["subcategory"].str.contains(health_concern, case=False, na=False)
            | filtered["name"].str.contains(health_concern, case=False, na=False)
        ]
        if not concern_filtered.empty:
            filtered = concern_filtered

    # Check current availability from demand CSV
    latest_date = dem["date"].max()
    latest_inv = dem[dem["date"] == latest_date][
        ["sku_id", "inventory"]
    ].drop_duplicates("sku_id")
    filtered = filtered.merge(latest_inv, on="sku_id", how="left")
    filtered["inventory"] = filtered["inventory"].fillna(0)

    # Sort by margin descending (best products for Pet Store first)
    filtered = filtered.sort_values("margin_pct", ascending=False).head(8)

    scope_parts = [f"Pet: {pet_type}"]
    if breed:
        scope_parts.append(f"Breed: {breed}")
    if age_months is not None:
        scope_parts.append(f"Age: {age_months} months ({life_stage})")
    if health_concern:
        scope_parts.append(f"Health: {health_concern}")
    scope = " | ".join(scope_parts)

    lines = [f"=== Pet Store Product Recommendations — {scope} ===\n"]
    if filtered.empty:
        return f"No products matched for: {scope}. Try broadening the filters."

    for i, (_, r) in enumerate(filtered.iterrows(), 1):
        avail_status = (
            "In Stock"
            if int(r["inventory"]) > 50
            else "Low Stock"
            if int(r["inventory"]) > 0
            else "Out of Stock"
        )
        suitability = []
        if breed and "breed_suitability" in r and pd.notna(r.get("breed_suitability")):
            if breed.lower() in str(r["breed_suitability"]).lower():
                suitability.append(f"Breed-specific for {breed}")
            else:
                suitability.append("Suitable for all breeds")
        if age_months is not None:
            suitability.append(f"Formulated for {life_stage}s")
        if health_concern:
            suitability.append(f"Addresses {health_concern}")

        lines.append(
            f"{i}. {r['name']} ({r['brand']})\n"
            f"   SKU         : {r['sku_id']}\n"
            f"   Category    : {r['category']} / {r['subcategory']}\n"
            f"   Price       : ₹{float(r['price_inr']):,.0f} | Margin: {r['margin_pct']:.1f}%\n"
            f"   Life Stage  : {r.get('life_stage', 'N/A')}\n"
            f"   Availability: {avail_status} ({int(r['inventory'])} units)\n"
            f"   Why suitable: {'; '.join(suitability) if suitability else 'Good match based on pet type and category'}\n"
        )

    lines.append(
        "── Shopping Tip ──\n"
        "Visit the company's online store or your nearest pet store for personalised recommendations.\n"
        "Ask our pet specialists for breed-specific feeding guides and portion advice."
    )
    return "\n".join(lines)


# ── Store Inventory Breakdown Tool ───────────────────────────────────────────


async def tool_get_store_inventory_breakdown(
    sku_id: str | None = None,
    category: str | None = None,
    max_days_of_supply: float = 7.0,
    region: str | None = None,
    city: str | None = None,
    risk_status: str | None = None,
    top_n: int = 20,
) -> str:
    """
    Query store_daily_inventory table (live DB) to return per-store, per-location
    inventory status. Answers questions like:
      - "Which stores have less than 7 days of stock?"
      - "Where in Mumbai is Royal Canin running low?"
      - "Which North India stores are critical for dog food?"

    Filters: sku_id, category, max_days_of_supply, region, city, risk_status.
    Returns store name, city, state, region, store type, SKU, inventory, days of supply, risk.
    Falls back to CSV if DB unavailable.
    """
    # Build fully-parameterised WHERE clauses (no f-string interpolation into SQL)
    # BUG-004 fix: days_of_supply and LIMIT are now parameters, not f-string literals
    conditions = ["days_of_supply <= %s"]
    params: list = [float(max_days_of_supply)]

    if sku_id:
        conditions.append("sku_id = %s")
        params.append(sku_id.upper().strip())
    if category:
        conditions.append("category = %s")
        params.append(category)
    if region:
        conditions.append("region = %s")
        params.append(region)
    if city:
        conditions.append("city = %s")
        params.append(city)
    if risk_status:
        conditions.append("risk_status = %s")
        params.append(risk_status.upper().strip())

    where = " AND ".join(conditions)
    top_n_safe = max(1, min(int(top_n), 500))  # clamp: never more than 500 rows

    sql = f"""
        SELECT
            store_id, city, state, region, store_type,
            sku_id, name, category, brand,
            inventory, days_of_supply, risk_status,
            price_inr, lead_time_days, record_date
        FROM store_daily_inventory
        WHERE {where}
          AND record_date = (SELECT MAX(record_date) FROM store_daily_inventory)
        ORDER BY days_of_supply ASC,
                 CASE risk_status WHEN 'CRITICAL' THEN 0 WHEN 'WARNING' THEN 1 ELSE 2 END ASC
        LIMIT %s
    """
    params.append(top_n_safe)

    rows = None
    source = "unknown"

    # Try MySQL first (use module-level import guard, not inline import)
    try:
        mysql_c = get_session_mysql_creds() or _get_default_mysql_creds()
        if mysql_c.get("host") and MYSQL_AVAILABLE:
            conn = await aiomysql.connect(
                host=mysql_c.get("host", "localhost"),
                port=int(mysql_c.get("port", 3306)),
                user=mysql_c.get("user", "root"),
                password=mysql_c.get("password", ""),
                db=mysql_c.get("db", "pet_store_scm"),
                autocommit=True,
            )
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
            conn.close()
            source = "MySQL (live)"
    except Exception as exc:
        logger.warning(f"[store_inventory] MySQL failed: {exc}")

    # Try PostgreSQL fallback
    if rows is None:
        try:
            pg_c = get_session_pg_creds() or _get_default_pg_creds()
            if pg_c.get("host"):
                import asyncpg

                conn2 = await asyncpg.connect(
                    host=pg_c.get("host", "localhost"),
                    port=int(pg_c.get("port", 5432)),
                    user=pg_c.get("user", "postgres"),
                    password=pg_c.get("password", ""),
                    database=pg_c.get("db", "pet_store_scm"),
                )
                # asyncpg uses $1, $2, … placeholders — replace each %s in order
                pg_sql = sql
                for i in range(1, len(params) + 1):
                    pg_sql = pg_sql.replace("%s", f"${i}", 1)
                rows_raw = await conn2.fetch(pg_sql, *params)
                rows = [dict(r) for r in rows_raw]
                await conn2.close()
                source = "PostgreSQL (live)"
        except Exception as exc:
            logger.warning(f"[store_inventory] PostgreSQL failed: {exc}")

    # CSV fallback
    if rows is None:
        try:
            sdi_path = DATA_DIR / "store_daily_inventory.csv"
            if sdi_path.exists():
                sdi = pd.read_csv(sdi_path, parse_dates=["date"])
                sdi = sdi.rename(columns={"date": "record_date"})
                latest = sdi["record_date"].max()
                sdi = sdi[sdi["record_date"] == latest]
                sdi = sdi[sdi["days_of_supply"] <= max_days_of_supply]
                if sku_id:
                    sdi = sdi[sdi["sku_id"] == sku_id.upper()]
                if category:
                    sdi = sdi[sdi["category"] == category]
                if region:
                    sdi = sdi[sdi["region"] == region]
                if city:
                    sdi = sdi[sdi["city"] == city]
                if risk_status:
                    sdi = sdi[sdi["risk_status"] == risk_status.upper()]
                sdi = sdi.sort_values("days_of_supply").head(top_n)
                rows = sdi.to_dict("records")
                source = f"CSV cache ({latest.date() if hasattr(latest, 'date') else latest})"
        except Exception as exc:
            return f"Error: Could not query store inventory from DB or CSV: {exc}"

    if not rows:
        filters = []
        if sku_id:
            filters.append(f"SKU={sku_id}")
        if category:
            filters.append(f"category={category}")
        if region:
            filters.append(f"region={region}")
        if city:
            filters.append(f"city={city}")
        filter_str = ", ".join(filters) if filters else "all SKUs"
        return (
            f"✓ No stores have less than {max_days_of_supply} days of supply "
            f"for {filter_str}. All locations appear adequately stocked."
        )

    # ── Build output as a Markdown table so the LLM can render it directly ──
    # The table format prevents the LLM from collapsing per-store rows into a
    # paragraph summary — it reproduces the table verbatim in its response.

    from collections import Counter

    critical = [r for r in rows if r.get("risk_status") == "CRITICAL"]
    warning = [r for r in rows if r.get("risk_status") == "WARNING"]
    ok_low = [r for r in rows if r.get("risk_status") not in ("CRITICAL", "WARNING")]
    n_crit, n_warn = len(critical), len(warning)

    header = (
        f"## Store Inventory Breakdown — ≤{max_days_of_supply} Days of Supply\n"
        f"**Source:** {source} &nbsp;|&nbsp; "
        f"**{len(rows)} location(s) found** &nbsp;|&nbsp; "
        f"🔴 {n_crit} Critical &nbsp;|&nbsp; 🟡 {n_warn} Warning\n\n"
    )

    # Markdown table columns
    tbl_header = (
        "| Risk | Store ID | City | State | Region | Store Type "
        "| SKU | Product | Category | Inventory | Days of Supply | Lead Time |\n"
        "|------|----------|------|-------|--------|------------"
        "|-----|---------|----------|-----------|---------------|----------|\n"
    )

    risk_icon = {"CRITICAL": "🔴 CRITICAL", "WARNING": "🟡 WARNING"}

    def _tbl_row(r: dict) -> str:
        dos = float(r.get("days_of_supply", 0))
        inv = int(r.get("inventory", 0))
        lt = int(r.get("lead_time_days", 7))
        risk = risk_icon.get(str(r.get("risk_status", "")), "ℹ LOW")
        name = str(r.get("name", ""))[:30].replace("|", "/")
        return (
            f"| {risk} | {r.get('store_id', '')} | {r.get('city', '')} "
            f"| {r.get('state', '')} | {r.get('region', '')} | {r.get('store_type', '')} "
            f"| {r.get('sku_id', '')} | {name} | {r.get('category', '')} "
            f"| {inv:,} units | **{dos:.1f}d** | {lt}d |\n"
        )

    tbl_rows = "".join(_tbl_row(r) for r in rows)

    # City summary
    city_counts = Counter(r.get("city", "") for r in rows)
    city_summary = ""
    if len(city_counts) > 1:
        top_cities = ", ".join(f"**{c}** ({n})" for c, n in city_counts.most_common(5))
        city_summary = f"\n**Cities most affected:** {top_cities}\n"

    # SKU summary — distinct SKUs at risk
    sku_counts = Counter(r.get("sku_id", "") for r in critical)
    sku_summary = ""
    if sku_counts:
        top_skus = ", ".join(
            f"{sku} ({n} stores)" for sku, n in sku_counts.most_common(5)
        )
        sku_summary = f"\n**Critical SKUs across most stores:** {top_skus}\n"

    # Recommendation
    avg_lt = int(
        round(
            sum(r.get("lead_time_days", 7) for r in critical[:5])
            / max(len(critical[:5]), 1)
        )
    )
    rec = (
        f"\n**Recommended actions:**\n"
        f"1. Place emergency orders for all 🔴 CRITICAL stores immediately "
        f"(avg lead time {avg_lt} days — some may already be stocked out on delivery).\n"
        f"2. Issue replenishment requests for 🟡 WARNING stores within 48 hours.\n"
        f"3. Run `generate_purchase_order` to create a full PO grouped by supplier.\n"
    )

    return header + tbl_header + tbl_rows + city_summary + sku_summary + rec


def _get_default_mysql_creds() -> dict:
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "db": os.getenv("MYSQL_DB", "pet_store_scm"),
    }


def _get_default_pg_creds() -> dict:
    return {
        "host": os.getenv("PG_HOST", "localhost"),
        "port": int(os.getenv("PG_PORT", 5432)),
        "user": os.getenv("PG_USER", "postgres"),
        "password": os.getenv("PG_PASSWORD", ""),
        "db": os.getenv("PG_DB", "pet_store_scm"),
    }


# MCP Protocol Definitions

MCP_TOOLS = [
    {
        "name": "get_inventory_status",
        "description": (
            "Returns current inventory status for pet store SKUs. "
            "Shows inventory level, average daily demand, days of supply, lead time, "
            "and risk classification (CRITICAL/WARNING/OK). "
            "Pass sku_id for a specific SKU or leave empty for top-N at-risk SKUs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "SKU ID (e.g. 'DOG_001', 'CAT_006', 'MED_003'). Leave empty for top at-risk summary.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of at-risk SKUs to return when sku_id is not specified. Default: 10.",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "get_demand_forecast",
        "description": (
            "Returns a 30-day (or custom horizon) demand forecast for a specific SKU. "
            "Provides pessimistic (P10), expected (P50), and optimistic (P90) demand estimates, "
            "plus a reorder recommendation based on current inventory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "SKU ID to forecast (e.g. 'DOG_001').",
                },
                "horizon_days": {
                    "type": "integer",
                    "description": "Forecast horizon in days. Default: 30.",
                    "default": 30,
                },
            },
            "required": ["sku_id"],
        },
    },
    {
        "name": "query_mysql",
        "description": (
            "Executes a read-only SQL query against the MySQL pet_store_scm database. "
            "MySQL contains: products, stores, customers, promotions, daily_demand, sales_transactions, returns, supplier_performance, cold_chain tables. "
            "NOTE: date column in daily_demand is named 'record_date' not 'date'. "
            "Only SELECT/SHOW/DESCRIBE queries are allowed. "
            "If creds are not passed, credentials from the .env file are used."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A read-only SQL SELECT statement.",
                },
                "creds": {
                    "type": "object",
                    "description": "Optional MySQL credentials: {host, port, user, password, db}. Overrides .env values.",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "db": {"type": "string"},
                    },
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "query_postgres",
        "description": (
            "Executes a read-only SQL query against the PostgreSQL pet_store_scm database. "
            "PostgreSQL contains: sku_forecasts, inventory_alerts, demand_anomalies, "
            "monthly_kpis, supplier_risk_scores, agent_query_log tables. "
            "Only SELECT queries are allowed. "
            "If creds are not passed, credentials from the .env file are used."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A read-only SQL SELECT statement.",
                },
                "creds": {
                    "type": "object",
                    "description": "Optional PostgreSQL credentials: {host, port, user, password, db}. Overrides .env values.",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "db": {"type": "string"},
                    },
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "test_mysql_connection",
        "description": "Tests MySQL database connectivity with the given credentials. Returns connection status, server version, and database name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "creds": {
                    "type": "object",
                    "description": "MySQL credentials: {host, port, user, password, db}. Uses .env values if not provided.",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "db": {"type": "string"},
                    },
                },
            },
        },
    },
    {
        "name": "test_postgres_connection",
        "description": "Tests PostgreSQL database connectivity with the given credentials. Returns connection status, server version, and database name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "creds": {
                    "type": "object",
                    "description": "PostgreSQL credentials: {host, port, user, password, db}. Uses .env values if not provided.",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "db": {"type": "string"},
                    },
                },
            },
        },
    },
    {
        "name": "get_supplier_info",
        "description": (
            "Returns structured supplier information including on-time delivery rate, "
            "quality rating, lead time, minimum order quantity, and notes. "
            "Pass supplier_name for details on one supplier, or leave empty for all suppliers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "supplier_name": {
                    "type": "string",
                    "description": "Supplier name (e.g. 'PawsSupply Co', 'TreatWorld LLC'). Leave empty for all.",
                },
            },
        },
    },
    {
        "name": "get_knowledge_base",
        "description": (
            "Returns structured domain knowledge and policies for pet store supply chain management. "
            "Available topics: reorder_policy, safety_stock, flea_tick_seasonality, holiday_demand, "
            "supplier_risk, new_sku_ramp, cold_chain, regulatory, shrink_loss, kpi_targets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Knowledge base topic (e.g. 'reorder_policy', 'safety_stock', 'holiday_demand').",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "log_forecast_to_postgres",
        "description": (
            "Writes a forecast result to the PostgreSQL sku_forecasts table for audit and tracking."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string"},
                "p10_total": {"type": "number"},
                "p50_total": {"type": "number"},
                "p90_total": {"type": "number"},
                "p50_daily": {"type": "number"},
                "horizon_days": {"type": "integer", "default": 30},
                "forecast_source": {
                    "type": "string",
                    "default": "TFT",
                },
                "model_version": {"type": "string", "default": "v1.0"},
            },
            "required": ["sku_id", "p10_total", "p50_total", "p90_total", "p50_daily"],
        },
    },
    {
        "name": "create_inventory_alert",
        "description": "Creates an inventory alert in PostgreSQL when a critical/warning condition is detected.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string"},
                "alert_type": {
                    "type": "string",
                    "enum": ["CRITICAL", "WARNING", "STOCKOUT"],
                },
                "days_of_supply": {"type": "number"},
                "current_inventory": {"type": "integer"},
                "avg_daily_demand": {"type": "number"},
                "lead_time_days": {"type": "integer"},
                "recommended_action": {"type": "string"},
            },
            "required": [
                "sku_id",
                "alert_type",
                "days_of_supply",
                "current_inventory",
                "avg_daily_demand",
                "lead_time_days",
                "recommended_action",
            ],
        },
    },
    {
        "name": "get_active_alerts",
        "description": "Fetches all unresolved inventory alerts from PostgreSQL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max alerts to return. Default: 20.",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "get_monthly_kpis",
        "description": "Fetches aggregated monthly supply chain KPIs from PostgreSQL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "Filter by SKU ID. Leave empty for company-wide KPIs.",
                },
                "months": {
                    "type": "integer",
                    "description": "Number of months to return. Default: 6.",
                    "default": 6,
                },
            },
        },
    },
    # ── High-value operational tools ──────────────────────────────────────
    {
        "name": "get_stockout_risk",
        "description": (
            "Returns all SKUs that will run out of stock within N days based on "
            "current inventory ÷ average daily demand. Ranked by urgency with "
            "supplier and suggested order quantity. "
            "USE THIS for: 'what will stock out soon', 'stockout risk', "
            "'what needs urgent reordering in the next X days'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Horizon in days. Default: 14.",
                    "default": 14,
                },
            },
        },
    },
    {
        "name": "get_reorder_list",
        "description": (
            "Generates the complete purchase order list: every SKU at or below its "
            "reorder point, ordered by urgency, with supplier, order quantity, and "
            "estimated cost. Total spend is summarised at the top. "
            "USE THIS for: 'what do I need to order', 'generate a reorder list', "
            "'purchase order recommendations', 'what should procurement do this week'."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_demand_trends",
        "description": (
            "Identifies SKUs with significantly increasing or decreasing demand by "
            "comparing the first and second half of a time period. "
            "USE THIS for: 'which products are trending up/down', 'demand changes', "
            "'growing/declining SKUs', 'demand this month vs last month'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Analysis window in days. Default: 90.",
                    "default": 90,
                },
            },
        },
    },
    # ── Composite tools (answer complex questions in ONE call) ─────────────
    {
        "name": "get_regional_inventory",
        "description": (
            "Returns inventory status grouped by region with CRITICAL/WARNING/OK counts "
            "and at-risk SKU details. Optionally filter by category. "
            "USE THIS for any question about regional stock levels, 'which regions are at risk', "
            "or 'status of [category] by region'. Answers in ONE call — no need for SQL."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional product category filter, e.g. 'Dog Treats', 'Cat Food'. Leave empty for all.",
                },
            },
        },
    },
    {
        "name": "get_supply_chain_dashboard",
        "description": (
            "Returns a complete supply chain executive summary in ONE call: total SKU counts, "
            "risk breakdown by category, top critical SKUs with reorder quantities, and key actions. "
            "USE THIS for broad questions: 'overall status', 'what needs attention', "
            "'give me a full report', 'executive summary', 'what is most at risk'."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_sku_360",
        "description": (
            "Returns a complete 360° profile of one SKU: inventory, risk, 30/90-day demand, "
            "trend, supplier, reorder point, safety stock, and recommended order quantity. "
            "USE THIS when you need everything about a specific SKU in one call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string", "description": "SKU ID, e.g. 'DOG_001'."},
            },
            "required": ["sku_id"],
        },
    },
    {
        "name": "get_supplier_ranking",
        "description": (
            "Returns all suppliers ranked by performance: lead time, reliability grade, "
            "number of SKUs supplied, and at-risk SKU counts. "
            "USE THIS for questions about 'best/worst supplier', 'who to reorder from', "
            "or 'supplier performance comparison'. Answers in ONE call."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "compare_categories",
        "description": (
            "Returns a side-by-side comparison of ALL product categories: health score, "
            "SKU risk breakdown, average days of supply, and most urgent SKUs per category. "
            "USE THIS for 'compare all categories', 'which category is worst', "
            "or 'how is each product line doing'. Answers in ONE call."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "web_search",
        "description": (
            "Search the web using Google (via SerpAPI) with automatic DuckDuckGo fallback. "
            "Use this for ANY question requiring external / real-world knowledge: "
            "market prices, industry news, supplier disruptions, competitor benchmarks, "
            "best practices, regulatory updates, weather events affecting supply chains, "
            "or any general knowledge question not answerable from the database. "
            "Returns titles, URLs, and snippets from top results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for better results.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1–10). Default: 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "python_repl",
        "description": (
            "Execute Python code in a secure sandboxed environment with pandas, numpy, "
            "and the full supply-chain DataFrame pre-loaded as `df`. "
            "Use this for: data quality checks (df[df['demand'] < 0]), "
            "statistical analysis (df.groupby('category')['demand'].describe()), "
            "custom calculations, MAPE/MAE computation, correlation analysis, "
            "time-series decomposition, forecasting accuracy evaluation, "
            "or ANY analytical task that SQL cannot express easily. "
            "The DataFrame `df` has columns: date, sku_id, demand, inventory, "
            "lead_time_days, price_usd, category, subcategory, supplier, region, name. "
            "Returns stdout output and/or the last expression value."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Valid Python code to execute. Use print() for output. "
                        "The last expression's value is also returned automatically. "
                        "Example: df[df['demand'] < 0][['sku_id','date','demand']]"
                    ),
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "data_quality",
        "description": (
            "Run a comprehensive data quality audit and statistical profiling of the "
            "supply-chain dataset. Checks for: negative values, null/missing data, "
            "per-SKU statistical outliers (|z-score| > 3), demand spike anomalies "
            "(> 3× rolling mean), sudden inventory drops (> 80% single-day), "
            "duplicate records, and full statistical profile (mean/std/min/max). "
            "USE THIS for questions like 'are there bad values?', 'is the data clean?', "
            "'any anomalies?', 'unnatural values?', 'data health check'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "'daily_demand', 'skus', or 'all'. Default: 'all'.",
                    "default": "all",
                },
                "checks": {
                    "type": "string",
                    "description": (
                        "Comma-separated checks to run, or 'all'. "
                        "Options: negatives, nulls, outliers, profile, anomalies. "
                        "Default: 'all'."
                    ),
                    "default": "all",
                },
            },
        },
    },
    # ── New Tools (20) ───────────────────────────────────────────────────────
    {
        "name": "get_brand_performance",
        "description": (
            "Pet Store brand performance ranking. Loads sales transactions and computes: "
            "total revenue (₹INR), total units sold, avg margin %, return rate, and stockout days per brand. "
            "Pass brand name for a detailed view of one brand, or leave empty for top-N ranking. "
            "USE THIS for: 'which brand sells most', 'brand-wise revenue', 'best performing brand'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "brand": {
                    "type": "string",
                    "description": "Brand name filter (e.g. 'Royal Canin', 'Pedigree'). Leave empty for all.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top brands to return. Default: 10.",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "get_franchise_inventory_comparison",
        "description": (
            "Compare inventory health across Pet Store stores and regions. "
            "Shows avg inventory, days of supply, critical SKU counts per region/store type. "
            "Highlights which regions/stores are most at risk of stockouts. "
            "USE THIS for: 'how are stores doing', 'regional inventory status', 'franchise health'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Filter by region (e.g. 'North', 'South', 'East', 'West', 'Central'). Leave empty for all.",
                },
                "store_type": {
                    "type": "string",
                    "description": "Filter by store type (e.g. 'Flagship', 'Standard'). Leave empty for all.",
                },
            },
        },
    },
    {
        "name": "get_seasonal_demand_calendar",
        "description": (
            "Seasonal demand calendar with Indian festival overlay. "
            "Computes monthly demand indices (actual vs annual avg) by category. "
            "Overlays Diwali, Navratri, Holi, monsoon season, and other Indian events. "
            "Recommends which categories to pre-stock in coming months. "
            "USE THIS for: 'seasonal planning', 'festival demand', 'when to stock up', 'demand calendar'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter to a specific product category. Leave empty for all.",
                },
                "months_ahead": {
                    "type": "integer",
                    "description": "Number of upcoming months to show. Default: 3.",
                    "default": 3,
                },
            },
        },
    },
    {
        "name": "get_cold_chain_monitor",
        "description": (
            "Cold chain monitoring for Pet Store refrigerated products. "
            "Reports: temperature breaches in last 7 days, units at risk of expiry in coming days, "
            "total estimated waste value. Alerts on shelf life < 3 days. "
            "USE THIS for: 'cold chain status', 'temperature alerts', 'expiry risk', 'perishable inventory'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "Days ahead to check for expiry risk. Default: 7.",
                    "default": 7,
                },
            },
        },
    },
    {
        "name": "get_supplier_lead_time_tracker",
        "description": (
            "Track actual vs promised lead times for all Pet Store suppliers. "
            "Flags suppliers with on-time delivery < 90%. Shows trend over 6 months. "
            "USE THIS for: 'supplier delays', 'lead time performance', 'which suppliers are late', "
            "'on-time delivery tracker'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "supplier_name": {
                    "type": "string",
                    "description": "Supplier name for detailed view. Leave empty for all suppliers.",
                },
            },
        },
    },
    {
        "name": "get_return_rate_analysis",
        "description": (
            "Return rate analysis by category and brand. Flags SKUs with return rate > 5%. "
            "Shows top return reasons. Compares against 3% industry benchmark. "
            "USE THIS for: 'return rates', 'why products are returned', 'high return SKUs', "
            "'return rate by brand'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by product category. Leave empty for all.",
                },
                "brand": {
                    "type": "string",
                    "description": "Filter by brand name. Leave empty for all.",
                },
            },
        },
    },
    {
        "name": "get_dead_stock_analysis",
        "description": (
            "Identify dead stock: SKUs with near-zero demand for N days. "
            "Calculates locked capital value, monthly holding cost (2%/month), "
            "and recommended clearance discount to sell in 30 days. "
            "USE THIS for: 'slow moving inventory', 'dead stock', 'clearance needed', 'locked capital'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "days_no_movement": {
                    "type": "integer",
                    "description": "Days of near-zero demand to classify as dead stock. Default: 60.",
                    "default": 60,
                },
            },
        },
    },
    {
        "name": "get_competitive_price_analysis",
        "description": (
            "Compare Pet Store prices with competitors (Amazon.in, Flipkart) for given SKU or brand. "
            "Returns Pet Store price and generates web_search queries for competitor price lookup. "
            "USE THIS for: 'price comparison', 'competitor prices', 'are we priced right', 'price gap'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "SKU ID to analyse (e.g. 'SKU_001'). Leave empty for brand-level.",
                },
                "brand": {
                    "type": "string",
                    "description": "Brand name filter (e.g. 'Royal Canin'). Leave empty for specific SKU.",
                },
            },
        },
    },
    {
        "name": "get_new_product_launch_readiness",
        "description": (
            "Assess health of a new Pet Store product launch: demand ramp (first 30 vs next 30 days), "
            "early stockout count, fill rate, and overall launch health score (0-100). "
            "USE THIS for: 'how is new product doing', 'launch performance', 'new SKU health'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "SKU ID of the new product to assess.",
                },
            },
            "required": ["sku_id"],
        },
    },
    {
        "name": "get_customer_segmentation_insights",
        "description": (
            "Customer segment analysis for the Pet Store's 5000 customers. "
            "Shows: avg order value, purchase frequency, LTV, top categories, preferred channel per segment. "
            "Recommends inventory allocation per segment. "
            "USE THIS for: 'customer segments', 'who buys most', 'segment LTV', 'top customer profiles'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "string",
                    "description": "Customer segment filter (e.g. 'Premium', 'Budget', 'Regular'). Leave empty for all.",
                },
            },
        },
    },
    {
        "name": "generate_purchase_order",
        "description": (
            "Generate a complete Pet Store purchase order for all SKUs below reorder point. "
            "Calculates order qty (90-day forecast - inventory - safety stock), groups by supplier, "
            "and estimates total cost in ₹INR. "
            "USE THIS for: 'generate PO', 'what to order from suppliers', 'purchase order', "
            "'procurement list', 'reorder urgency'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urgency": {
                    "type": "string",
                    "description": "Filter by urgency: 'critical', 'warning', or 'all'. Default: 'all'.",
                    "default": "all",
                },
                "supplier_name": {
                    "type": "string",
                    "description": "Filter to a specific supplier. Leave empty for all suppliers.",
                },
            },
        },
    },
    {
        "name": "get_promotion_inventory_impact",
        "description": (
            "Analyse how Pet Store promotions affected demand and inventory. "
            "For each promo: demand lift %, stockout SKUs during promo, restock lag after. "
            "Identifies categories that benefited most vs suffered stockouts. "
            "USE THIS for: 'promo impact', 'did our sale cause stockouts', 'demand lift from promotion'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "promo_id": {
                    "type": "string",
                    "description": "Specific promotion ID. Leave empty for all promotions.",
                },
                "days_before": {
                    "type": "integer",
                    "description": "Baseline days before promo start. Default: 7.",
                    "default": 7,
                },
                "days_after": {
                    "type": "integer",
                    "description": "Days after promo end to measure restock lag. Default: 14.",
                    "default": 14,
                },
            },
        },
    },
    {
        "name": "get_channel_revenue_attribution",
        "description": (
            "Revenue breakdown by sales channel (Online/Offline/App) for the Pet Store. "
            "Shows: revenue share, units, margin %, avg order value, top SKUs per channel. "
            "USE THIS for: 'channel performance', 'online vs offline sales', 'which channel drives most revenue'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Filter to specific channel (e.g. 'Online', 'Offline', 'App'). Leave empty for all.",
                },
                "period": {
                    "type": "string",
                    "description": "'last_90_days', 'last_30_days', or 'all'. Default: 'last_90_days'.",
                    "default": "last_90_days",
                },
            },
        },
    },
    {
        "name": "get_markdown_optimization",
        "description": (
            "Identify overstocked Pet Store SKUs and generate optimal clearance discounts. "
            "Flags SKUs with inventory > 3× 90-day demand. Calculates: days to clear, "
            "suggested discount %, revenue at discount vs holding cost savings. "
            "USE THIS for: 'overstocked products', 'clearance pricing', 'markdown recommendations', "
            "'slow sellers'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter to a specific product category. Leave empty for all.",
                },
            },
        },
    },
    {
        "name": "get_marketing_campaign_recommendations",
        "description": (
            "Identify top 5 Pet Store categories to promote NOW (overstocked or high margin) "
            "and top 5 to AVOID promoting (understocked — would cause stockouts). "
            "Cross-references seasonal demand calendar for upcoming peaks. "
            "USE THIS for: 'what to promote', 'marketing priorities', 'campaign planning', "
            "'which categories need a push'."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_inventory_financial_summary",
        "description": (
            "CFO-level inventory financial report for the Pet Store. "
            "Shows: total inventory value (cost), retail value, potential gross margin, "
            "dead stock value, stockout lost revenue (30d), working capital days. "
            "All values in ₹INR. "
            "USE THIS for: 'inventory financial report', 'working capital', 'inventory value', "
            "'financial summary', 'how much is inventory worth'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "'current' or date string. Default: 'current'.",
                    "default": "current",
                },
            },
        },
    },
    {
        "name": "get_customer_cohort_demand_analysis",
        "description": (
            "Customer cohort analysis: group Pet Store customers by join date (quarterly). "
            "Shows avg LTV, order frequency, top pet types and channels per cohort. "
            "Identifies which cohorts have highest LTV and what they buy. "
            "USE THIS for: 'cohort analysis', 'customer retention', 'LTV by cohort', "
            "'which customers are most valuable'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cohort_months": {
                    "type": "integer",
                    "description": "Cohort period in months (e.g. 3 = quarterly). Default: 3.",
                    "default": 3,
                },
            },
        },
    },
    {
        "name": "get_store_level_demand_intelligence",
        "description": (
            "Store-level demand intelligence for the Pet Store's 67 stores across India. "
            "Identifies stores with unique demand vs national average, consistently under/overstocked stores. "
            "Shows top SKUs per store and inventory rebalancing opportunities. "
            "USE THIS for: 'store performance', 'city-wise demand', 'which city sells most', "
            "'store-level insights', 'rebalance inventory'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "string",
                    "description": "Filter to specific store ID. Leave empty for all.",
                },
                "city": {
                    "type": "string",
                    "description": "Filter to specific city (e.g. 'Mumbai', 'Delhi', 'Bangalore'). Leave empty for all.",
                },
            },
        },
    },
    {
        "name": "get_supplier_negotiation_brief",
        "description": (
            "Supplier negotiation brief with leverage score (0-10) and talking points. "
            "Shows: YoY volume growth, lead time vs contract, OTD trend, stockout incidents, "
            "defect rate, and specific negotiation arguments. "
            "USE THIS for: 'negotiation strategy', 'supplier leverage', 'how to negotiate with supplier', "
            "'supplier talking points', 'annual vendor review'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "supplier_name": {
                    "type": "string",
                    "description": "Supplier name for focused brief. Leave empty for all suppliers.",
                },
            },
        },
    },
    {
        "name": "get_product_recommendation",
        "description": (
            "Recommend Pet Store products for a specific pet based on type, breed, age, and health concern. "
            "Filters by: pet_type (Dog/Cat/Bird/Fish), breed suitability, life_stage (Puppy/Adult/Senior "
            "derived from age_months: 0-12=Puppy/Kitten, 12-84=Adult, 84+=Senior), and health concern. "
            "Returns top products with: name, brand, price (₹INR), suitability reason, availability. "
            "USE THIS for: 'what food for Labrador puppy', 'products for senior cat', "
            "'recommendations for 4-month-old dog', 'tick flea products for dog', "
            "'what should I buy for my pet'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pet_type": {
                    "type": "string",
                    "description": "Pet type: 'Dog', 'Cat', 'Bird', 'Fish', 'Small Animal'. Required.",
                },
                "breed": {
                    "type": "string",
                    "description": "Breed name (e.g. 'Labrador Retriever', 'German Shepherd', 'Persian'). Optional.",
                },
                "age_months": {
                    "type": "integer",
                    "description": "Pet's age in months (e.g. 4 for a 4-month-old puppy). Optional.",
                },
                "health_concern": {
                    "type": "string",
                    "description": "Health concern or special need (e.g. 'tick flea', 'joint', 'dental', 'sensitive stomach'). Optional.",
                },
            },
            "required": ["pet_type"],
        },
    },
    {
        "name": "get_store_inventory_breakdown",
        "description": (
            "Query the store_daily_inventory table (live database) to get per-store, "
            "per-location inventory levels with exact store names, cities, states, "
            "and regions. USE THIS whenever the user asks about specific store locations, "
            "cities, or 'where' a stockout is happening. "
            "Answers: 'Which stores have less than 7 days of stock?', "
            "'Where in Mumbai is Royal Canin running low?', "
            "'Which North India stores are critical for dog food?', "
            "'Pinpoint the exact location of stock shortages'. "
            "Returns store ID, city, state, region, store type, SKU, inventory, "
            "days of supply, and risk status. Queries live MySQL/PostgreSQL first, "
            "falls back to CSV cache. Always use this instead of get_inventory_status "
            "when location-specific answers are needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "Filter to a specific SKU ID (e.g. 'FOOD_D001'). Optional.",
                },
                "category": {
                    "type": "string",
                    "description": "Filter to a product category (e.g. 'Dog Food', 'Health'). Optional.",
                },
                "max_days_of_supply": {
                    "type": "number",
                    "description": "Return stores with LESS than this many days of supply. Default: 7.",
                    "default": 7.0,
                },
                "region": {
                    "type": "string",
                    "description": "Filter to a region: 'North', 'South', 'East', 'West'. Optional.",
                },
                "city": {
                    "type": "string",
                    "description": "Filter to a specific city (e.g. 'Mumbai', 'Delhi', 'Bengaluru'). Optional.",
                },
                "risk_status": {
                    "type": "string",
                    "description": "Filter by risk: 'CRITICAL', 'WARNING', or 'OK'. Optional.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Maximum number of store-SKU combinations to return. Default: 20.",
                    "default": 20,
                },
            },
        },
    },
    # ── New tools ──────────────────────────────────────────────────────────────
    {
        "name": "get_transfer_recommendations",
        "description": (
            "Identifies inter-store stock transfer opportunities: overstocked stores "
            "that can donate inventory to critically understocked stores for the same SKU, "
            "avoiding emergency purchase orders. Returns a Markdown table of recommended "
            "transfers with quantities, urgency, and store locations. "
            "USE THIS for: 'should we move stock between stores?', 'which stores can help critical ones?', "
            "'avoid emergency POs by transferring surplus stock'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_days_of_supply": {
                    "type": "number",
                    "description": "Threshold for 'needs stock' (default 7 days).",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max transfer recommendations to return (default 15).",
                },
            },
        },
    },
    {
        "name": "get_abc_xyz_analysis",
        "description": (
            "ABC-XYZ inventory classification of all SKUs. "
            "ABC = revenue contribution (A=top 70%, B=next 20%, C=bottom 10%). "
            "XYZ = demand variability (X=stable CV<0.5, Y=variable, Z=erratic CV>1.0). "
            "Returns a ranked Markdown table with class, revenue, coefficient of variation, "
            "and recommended stocking strategy per SKU. "
            "USE THIS for: 'which are our most important SKUs?', 'ABC analysis', "
            "'which products should always be in stock?', 'inventory prioritisation'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter to a specific product category. Optional.",
                },
            },
        },
    },
    {
        "name": "get_supplier_fill_rate_trend",
        "description": (
            "Shows trend of each supplier's OTD%, fill rate%, and defect rate over the last N months. "
            "Identifies suppliers getting worse (📉) vs improving (📈) with slope values. "
            "Returns a Markdown table sorted by worst trend first. "
            "USE THIS for: 'which suppliers are getting worse?', 'how has supplier X's performance changed?', "
            "'supplier trend analysis', 'supplier performance over time'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "supplier_name": {
                    "type": "string",
                    "description": "Filter to a specific supplier. Optional.",
                },
                "months": {
                    "type": "integer",
                    "description": "Number of months to analyse (default 6).",
                },
            },
        },
    },
    {
        "name": "get_basket_analysis",
        "description": (
            "Market basket analysis: identifies products most frequently bought together "
            "in the same transaction. Computes co-purchase counts and support percentage. "
            "Returns a Markdown table of top product pairs with bundle opportunity scores. "
            "USE THIS for: 'what do customers buy together?', 'cross-sell opportunities', "
            "'which products should we bundle?', 'shelf placement recommendations'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter to a product category. Optional.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Top N pairs to return (default 15).",
                },
            },
        },
    },
    {
        "name": "get_price_elasticity_analysis",
        "description": (
            "Estimates price elasticity of demand using historical promotion data. "
            "Shows how much demand changes per 1% discount. Classifies as: "
            "Elastic (>1.5), Moderate (0.5-1.5), Inelastic (<0.5), Negative (premium effect). "
            "Returns a Markdown table per promotion event with elasticity score. "
            "USE THIS for: 'how sensitive is demand to price?', 'which products respond to discounts?', "
            "'should we run a sale on X?', 'price elasticity', 'discount impact analysis'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter to a product category. Optional.",
                },
            },
        },
    },
    {
        "name": "get_forecast_vs_actual",
        "description": (
            "Compares forecasted demand (from MLOps prediction log) against actual demand "
            "for the same period. Computes MAPE, bias (over/under forecast), and grades "
            "each SKU A-D. Flags SKUs needing model retraining. "
            "USE THIS for: 'how accurate were our forecasts?', 'which SKUs have worst forecast error?', "
            "'forecast accuracy report', 'MAPE by SKU', 'model performance'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "Filter to a specific SKU. Optional.",
                },
                "days": {
                    "type": "integer",
                    "description": "Lookback days for actual demand (default 30).",
                },
            },
        },
    },
]

# ── Auto-populate _TOOL_REGISTRY from the static MCP_TOOLS list ──────────────
# This bridges the old static list with the new registry pattern.
# Tools that only read CSV data are marked cacheable (TTL = 60 s by default).
# DB-writing and DB-reading tools are NOT cacheable (always hit the live DB).
# The credential-injection tools (query_mysql, query_postgres, etc.) are handled
# via a special wrapper below rather than being registered as plain callables.

# Pure read-only CSV tools — safe to cache
_CACHEABLE_TOOLS: set[str] = {
    "get_inventory_status",
    "get_stockout_risk",
    "get_reorder_list",
    "get_demand_trends",
    "get_regional_inventory",
    "get_supply_chain_dashboard",
    "get_sku_360",
    "get_supplier_ranking",
    "compare_categories",
    "data_quality",
    "get_brand_performance",
    "get_franchise_inventory_comparison",
    "get_seasonal_demand_calendar",
    "get_cold_chain_monitor",
    "get_supplier_lead_time_tracker",
    "get_return_rate_analysis",
    "get_dead_stock_analysis",
    "get_competitive_price_analysis",
    "get_new_product_launch_readiness",
    "get_customer_segmentation_insights",
    "generate_purchase_order",
    "get_promotion_inventory_impact",
    "get_channel_revenue_attribution",
    "get_markdown_optimization",
    "get_marketing_campaign_recommendations",
    "get_inventory_financial_summary",
    "get_customer_cohort_demand_analysis",
    "get_store_level_demand_intelligence",
    "get_supplier_negotiation_brief",
    "get_product_recommendation",
    "get_supplier_info",
    "get_knowledge_base",
    "get_demand_forecast",
    # New tools
    "get_transfer_recommendations",
    "get_abc_xyz_analysis",
    "get_supplier_fill_rate_trend",
    "get_basket_analysis",
    "get_price_elasticity_analysis",
    "get_forecast_vs_actual",
}

# DB tools that need credential injection — handled specially in dispatch
_DB_TOOLS: set[str] = {
    "query_mysql",
    "query_postgres",
    "test_mysql_connection",
    "test_postgres_connection",
    "log_forecast_to_postgres",
    "create_inventory_alert",
    "get_active_alerts",
    "get_monthly_kpis",
    "get_store_inventory_breakdown",
}

# Map tool name → implementation function (for registry population)
_TOOL_HANDLER_MAP: dict[str, Any] = {
    "get_inventory_status": lambda **kw: tool_get_inventory_status(**kw),
    "get_demand_forecast": lambda **kw: tool_get_demand_forecast(**kw),
    "get_supplier_info": lambda **kw: tool_get_supplier_info(**kw),
    "get_knowledge_base": lambda **kw: tool_get_knowledge_base(**kw),
    "get_stockout_risk": lambda **kw: tool_get_stockout_risk(**kw),
    "get_reorder_list": lambda **kw: tool_get_reorder_list(),
    "get_demand_trends": lambda **kw: tool_get_demand_trends(**kw),
    "get_regional_inventory": lambda **kw: tool_get_regional_inventory(**kw),
    "get_supply_chain_dashboard": lambda **kw: tool_get_supply_chain_dashboard(),
    "get_sku_360": lambda **kw: tool_get_sku_360(**kw),
    "get_supplier_ranking": lambda **kw: tool_get_supplier_ranking(),
    "compare_categories": lambda **kw: tool_compare_categories(),
    "web_search": lambda **kw: tool_web_search(**kw),
    "python_repl": lambda **kw: tool_python_repl(**kw),
    "data_quality": lambda **kw: tool_data_quality(**kw),
    "get_brand_performance": lambda **kw: tool_get_brand_performance(**kw),
    "get_franchise_inventory_comparison": lambda **kw: (
        tool_get_franchise_inventory_comparison(**kw)
    ),
    "get_seasonal_demand_calendar": lambda **kw: tool_get_seasonal_demand_calendar(
        **kw
    ),
    "get_cold_chain_monitor": lambda **kw: tool_get_cold_chain_monitor(**kw),
    "get_supplier_lead_time_tracker": lambda **kw: tool_get_supplier_lead_time_tracker(
        **kw
    ),
    "get_return_rate_analysis": lambda **kw: tool_get_return_rate_analysis(**kw),
    "get_dead_stock_analysis": lambda **kw: tool_get_dead_stock_analysis(**kw),
    "get_competitive_price_analysis": lambda **kw: tool_get_competitive_price_analysis(
        **kw
    ),
    "get_new_product_launch_readiness": lambda **kw: (
        tool_get_new_product_launch_readiness(**kw)
    ),
    "get_customer_segmentation_insights": lambda **kw: (
        tool_get_customer_segmentation_insights(**kw)
    ),
    "generate_purchase_order": lambda **kw: tool_generate_purchase_order(**kw),
    "get_promotion_inventory_impact": lambda **kw: tool_get_promotion_inventory_impact(
        **kw
    ),
    "get_channel_revenue_attribution": lambda **kw: (
        tool_get_channel_revenue_attribution(**kw)
    ),
    "get_markdown_optimization": lambda **kw: tool_get_markdown_optimization(**kw),
    "get_marketing_campaign_recommendations": lambda **kw: (
        tool_get_marketing_campaign_recommendations(**kw)
    ),
    "get_inventory_financial_summary": lambda **kw: (
        tool_get_inventory_financial_summary(**kw)
    ),
    "get_customer_cohort_demand_analysis": lambda **kw: (
        tool_get_customer_cohort_demand_analysis(**kw)
    ),
    "get_store_level_demand_intelligence": lambda **kw: (
        tool_get_store_level_demand_intelligence(**kw)
    ),
    "get_supplier_negotiation_brief": lambda **kw: tool_get_supplier_negotiation_brief(
        **kw
    ),
    "get_product_recommendation": lambda **kw: tool_get_product_recommendation(**kw),
    # New tools
    "get_transfer_recommendations": lambda **kw: tool_get_transfer_recommendations(
        **kw
    ),
    "get_abc_xyz_analysis": lambda **kw: tool_get_abc_xyz_analysis(**kw),
    "get_supplier_fill_rate_trend": lambda **kw: tool_get_supplier_fill_rate_trend(
        **kw
    ),
    "get_basket_analysis": lambda **kw: tool_get_basket_analysis(**kw),
    "get_price_elasticity_analysis": lambda **kw: tool_get_price_elasticity_analysis(
        **kw
    ),
    "get_forecast_vs_actual": lambda **kw: tool_get_forecast_vs_actual(**kw),
}


def _build_tool_registry() -> None:
    """
    Populate _TOOL_REGISTRY from the static MCP_TOOLS list + _TOOL_HANDLER_MAP.
    Called once after all tool functions are defined (end of module).
    DB tools that need credential injection are NOT registered here — they are
    handled directly in dispatch_tool() which has access to session creds.
    """
    for schema_entry in MCP_TOOLS:
        name = schema_entry["name"]
        if name in _DB_TOOLS:
            # DB tools stay in the old dispatch path for credential injection
            continue
        handler = _TOOL_HANDLER_MAP.get(name)
        if handler is None:
            continue
        _TOOL_REGISTRY[name] = {
            "schema": schema_entry,
            "handler": handler,
            "is_async": False,  # CSV tools are sync
            "cacheable": name in _CACHEABLE_TOOLS,
            "cache_ttl": _TOOL_CACHE_DEFAULT_TTL,
        }


MCP_RESOURCES = [
    {
        "uri": "pet-store://policies/reorder",
        "name": "Reorder Policy",
        "description": "Pet store reorder point policies by category",
        "mimeType": "text/plain",
    },
    {
        "uri": "pet-store://policies/safety-stock",
        "name": "Safety Stock Guidelines",
        "description": "Safety stock formula and service level targets",
        "mimeType": "text/plain",
    },
    {
        "uri": "pet-store://suppliers/all",
        "name": "All Supplier Profiles",
        "description": "Performance data for all 9 pet store suppliers",
        "mimeType": "text/plain",
    },
    {
        "uri": "pet-store://sku-catalog",
        "name": "SKU Catalog",
        "description": "Full catalog of 65 pet store SKUs",
        "mimeType": "text/plain",
    },
]

# MCP Message Handlers


async def handle_mcp_message(message: dict) -> dict:
    """Process a single JSON-RPC 2.0 MCP message and return a response."""
    msg_id = message.get("id")
    method = message.get("method", "")
    params = message.get("params", {})

    def ok(result):
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def err(code, msg):
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": msg}}

    try:
        # initialization
        if method == "initialize":
            return ok(
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "pet-store-scm-mcp",
                        "version": "1.0.0",
                    },
                }
            )

        elif method == "initialized":
            return None  # notification, no response

        # tools
        elif method == "tools/list":
            return ok({"tools": MCP_TOOLS})

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            result_text = await dispatch_tool(tool_name, args)
            return ok(
                {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": result_text.startswith("ERROR"),
                }
            )

        # resources
        elif method == "resources/list":
            return ok({"resources": MCP_RESOURCES})

        elif method == "resources/read":
            uri = params.get("uri", "")
            content = await read_resource(uri)
            return ok(
                {"contents": [{"uri": uri, "mimeType": "text/plain", "text": content}]}
            )

        # ping
        elif method == "ping":
            return ok({})

        else:
            return err(-32601, f"Method not found: {method}")

    except Exception as exc:
        traceback.print_exc()
        return err(-32603, f"Internal error: {exc}")


async def dispatch_tool(name: str, args: dict) -> str:
    """
    Route a tool call to its implementation.

    Routing strategy (two paths):
    1. Non-DB tools registered in _TOOL_REGISTRY → _call_registered_tool()
       These benefit from TTL caching (cacheable=True tools) and structured
       error messages with self-correction hints built in.
    2. DB tools in _DB_TOOLS → _dispatch_tool_inner() for credential injection
       These require the session ContextVar creds set by set_session_creds().

    All errors are caught and returned as structured, human-readable messages
    so the agent can reason about what went wrong and try an alternative approach
    rather than receiving a raw Python traceback.
    """
    if name in _TOOL_REGISTRY:
        # Fast path — registry-based dispatch with TTL caching
        return await _call_registered_tool(name, args)
    # Credential-injecting DB tools fall through to the legacy inner dispatch
    try:
        return await _dispatch_tool_inner(name, args)
    except Exception as exc:
        tb_short = traceback.format_exc(limit=3)
        return (
            f"TOOL_ERROR [{name}]: {type(exc).__name__}: {exc}\n\n"
            f"What this means:\n"
            f"  - The tool '{name}' failed to execute.\n"
            f"  - Possible causes: database not connected, bad SQL syntax, "
            f"missing data, or invalid arguments.\n\n"
            f"Suggested alternatives:\n"
            f"  - If this was a SQL query: try python_repl with the same logic on `df`\n"
            f"  - If this was a DB tool: check database connection with "
            f"test_mysql_connection or test_postgres_connection\n"
            f"  - If data is unavailable: answer from CSV cache using python_repl or "
            f"built-in inventory tools\n\n"
            f"Technical detail (for debugging):\n{tb_short}"
        )


async def _dispatch_tool_inner(name: str, args: dict) -> str:
    """Inner dispatch — actual tool routing. Wrapped by dispatch_tool for error handling."""
    # If the LLM didn't pass explicit creds in the tool call args,
    # fall back to the session creds pushed by the UI before this chat turn.
    _effective_mysql = (
        (
            args.get("creds")
            if name in ("query_mysql", "test_mysql_connection")
            else None
        )
        or get_session_mysql_creds()
        or None
    )

    _pg_tools = (
        "query_postgres",
        "test_postgres_connection",
        "log_forecast_to_postgres",
        "create_inventory_alert",
        "get_active_alerts",
        "get_monthly_kpis",
    )
    _effective_pg = (
        (args.get("creds") if name in _pg_tools else None)
        or get_session_pg_creds()
        or None
    )

    if name == "get_inventory_status":
        return tool_get_inventory_status(
            sku_id=args.get("sku_id"),
            top_n=args.get("top_n", 10),
        )
    elif name == "get_demand_forecast":
        return tool_get_demand_forecast(
            sku_id=args["sku_id"],
            horizon_days=args.get("horizon_days", 30),
        )
    elif name == "query_mysql":
        return await tool_query_mysql(args["sql"], creds=_effective_mysql)
    elif name == "query_postgres":
        return await tool_query_postgres(args["sql"], creds=_effective_pg)
    elif name == "test_mysql_connection":
        return await tool_test_mysql_connection(creds=_effective_mysql)
    elif name == "test_postgres_connection":
        return await tool_test_postgres_connection(creds=_effective_pg)
    elif name == "get_supplier_info":
        return tool_get_supplier_info(args.get("supplier_name"))
    elif name == "get_knowledge_base":
        return tool_get_knowledge_base(args["topic"])
    elif name == "log_forecast_to_postgres":
        return await tool_log_forecast_to_postgres(
            sku_id=args["sku_id"],
            p10_total=args["p10_total"],
            p50_total=args["p50_total"],
            p90_total=args["p90_total"],
            p50_daily=args["p50_daily"],
            horizon_days=args.get("horizon_days", 30),
            forecast_source=args.get("forecast_source", "TFT"),
            model_version=args.get("model_version", "v1.0"),
        )
    elif name == "create_inventory_alert":
        return await tool_create_inventory_alert(
            sku_id=args["sku_id"],
            alert_type=args["alert_type"],
            days_of_supply=args["days_of_supply"],
            current_inventory=args["current_inventory"],
            avg_daily_demand=args["avg_daily_demand"],
            lead_time_days=args["lead_time_days"],
            recommended_action=args["recommended_action"],
        )
    elif name == "get_active_alerts":
        return await tool_get_active_alerts(limit=args.get("limit", 20))
    elif name == "get_monthly_kpis":
        return await tool_get_monthly_kpis(
            sku_id=args.get("sku_id"),
            months=args.get("months", 6),
        )
    elif name == "get_stockout_risk":
        return tool_get_stockout_risk(days=args.get("days", 14))
    elif name == "get_reorder_list":
        return tool_get_reorder_list()
    elif name == "get_demand_trends":
        return tool_get_demand_trends(days=args.get("days", 90))
    elif name == "get_regional_inventory":
        return tool_get_regional_inventory(category=args.get("category"))
    elif name == "get_supply_chain_dashboard":
        return tool_get_supply_chain_dashboard()
    elif name == "get_sku_360":
        return tool_get_sku_360(sku_id=args["sku_id"])
    elif name == "get_supplier_ranking":
        return tool_get_supplier_ranking()
    elif name == "compare_categories":
        return tool_compare_categories()
    elif name == "web_search":
        return tool_web_search(
            query=args["query"],
            num_results=args.get("num_results", 5),
        )
    elif name == "python_repl":
        return tool_python_repl(code=args["code"])
    elif name == "data_quality":
        return tool_data_quality(
            table=args.get("table", "all"),
            checks=args.get("checks", "all"),
        )
    # ── New Tools ────────────────────────────────────────────────────────────
    elif name == "get_brand_performance":
        return tool_get_brand_performance(
            brand=args.get("brand"),
            top_n=args.get("top_n", 10),
        )
    elif name == "get_franchise_inventory_comparison":
        return tool_get_franchise_inventory_comparison(
            region=args.get("region"),
            store_type=args.get("store_type"),
        )
    elif name == "get_seasonal_demand_calendar":
        return tool_get_seasonal_demand_calendar(
            category=args.get("category"),
            months_ahead=args.get("months_ahead", 3),
        )
    elif name == "get_cold_chain_monitor":
        return tool_get_cold_chain_monitor(
            days_ahead=args.get("days_ahead", 7),
        )
    elif name == "get_supplier_lead_time_tracker":
        return tool_get_supplier_lead_time_tracker(
            supplier_name=args.get("supplier_name"),
        )
    elif name == "get_return_rate_analysis":
        return tool_get_return_rate_analysis(
            category=args.get("category"),
            brand=args.get("brand"),
        )
    elif name == "get_dead_stock_analysis":
        return tool_get_dead_stock_analysis(
            days_no_movement=args.get("days_no_movement", 60),
        )
    elif name == "get_competitive_price_analysis":
        return tool_get_competitive_price_analysis(
            sku_id=args.get("sku_id"),
            brand=args.get("brand"),
        )
    elif name == "get_new_product_launch_readiness":
        return tool_get_new_product_launch_readiness(
            sku_id=args["sku_id"],
        )
    elif name == "get_customer_segmentation_insights":
        return tool_get_customer_segmentation_insights(
            segment=args.get("segment"),
        )
    elif name == "generate_purchase_order":
        return tool_generate_purchase_order(
            urgency=args.get("urgency", "all"),
            supplier_name=args.get("supplier_name"),
        )
    elif name == "get_promotion_inventory_impact":
        return tool_get_promotion_inventory_impact(
            promo_id=args.get("promo_id"),
            days_before=args.get("days_before", 7),
            days_after=args.get("days_after", 14),
        )
    elif name == "get_channel_revenue_attribution":
        return tool_get_channel_revenue_attribution(
            channel=args.get("channel"),
            period=args.get("period", "last_90_days"),
        )
    elif name == "get_markdown_optimization":
        return tool_get_markdown_optimization(
            category=args.get("category"),
        )
    elif name == "get_marketing_campaign_recommendations":
        return tool_get_marketing_campaign_recommendations()
    elif name == "get_inventory_financial_summary":
        return tool_get_inventory_financial_summary(
            period=args.get("period", "current"),
        )
    elif name == "get_customer_cohort_demand_analysis":
        return tool_get_customer_cohort_demand_analysis(
            cohort_months=args.get("cohort_months", 3),
        )
    elif name == "get_store_level_demand_intelligence":
        return tool_get_store_level_demand_intelligence(
            store_id=args.get("store_id"),
            city=args.get("city"),
        )
    elif name == "get_supplier_negotiation_brief":
        return tool_get_supplier_negotiation_brief(
            supplier_name=args.get("supplier_name"),
        )
    elif name == "get_product_recommendation":
        return tool_get_product_recommendation(
            pet_type=args["pet_type"],
            breed=args.get("breed"),
            age_months=args.get("age_months"),
            health_concern=args.get("health_concern"),
        )
    elif name == "get_store_inventory_breakdown":
        return await tool_get_store_inventory_breakdown(
            sku_id=args.get("sku_id"),
            category=args.get("category"),
            max_days_of_supply=float(args.get("max_days_of_supply", 7.0)),
            region=args.get("region"),
            city=args.get("city"),
            risk_status=args.get("risk_status"),
            top_n=int(args.get("top_n", 20)),
        )
    elif name == "get_transfer_recommendations":
        return tool_get_transfer_recommendations(
            max_days_of_supply=float(args.get("max_days_of_supply", 7.0)),
            top_n=int(args.get("top_n", 15)),
        )
    elif name == "get_abc_xyz_analysis":
        return tool_get_abc_xyz_analysis(category=args.get("category"))
    elif name == "get_supplier_fill_rate_trend":
        return tool_get_supplier_fill_rate_trend(
            supplier_name=args.get("supplier_name"),
            months=int(args.get("months", 6)),
        )
    elif name == "get_basket_analysis":
        return tool_get_basket_analysis(
            category=args.get("category"),
            top_n=int(args.get("top_n", 15)),
        )
    elif name == "get_price_elasticity_analysis":
        return tool_get_price_elasticity_analysis(category=args.get("category"))
    elif name == "get_forecast_vs_actual":
        return tool_get_forecast_vs_actual(
            sku_id=args.get("sku_id"),
            days=int(args.get("days", 30)),
        )
    else:
        return (
            f"TOOL_ERROR: Unknown tool '{name}'. "
            f"Available tools: {[t['name'] for t in MCP_TOOLS]}"
        )


async def read_resource(uri: str) -> str:
    if uri == "pet-store://policies/reorder":
        return tool_get_knowledge_base("reorder_policy")
    elif uri == "pet-store://policies/safety-stock":
        return tool_get_knowledge_base("safety_stock")
    elif uri == "pet-store://suppliers/all":
        return tool_get_supplier_info()
    elif uri == "pet-store://sku-catalog":
        df = get_df()
        latest = (
            df[df["date"] == df["date"].max()][
                [
                    "sku_id",
                    "name",
                    "category",
                    "subcategory",
                    "supplier",
                    "region",
                    "lead_time_days",
                    "price_usd",
                ]
            ]
            .drop_duplicates("sku_id")
            .sort_values("sku_id")
        )
        return latest.to_string(index=False)
    return f"Unknown resource URI: {uri}"


# FastAPI Application


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Pre-load CSV at startup
    try:
        get_df()
        print(f"[MCP] CSV loaded: {len(_df_cache):,} rows")
    except FileNotFoundError as e:
        print(f"[MCP] WARNING: {e}")
    yield


app = FastAPI(
    title="Pet Store SCM – MCP Server",
    description="Model Context Protocol server for pet store supply chain intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow only explicitly configured origins.
# In production set MCP_ALLOWED_ORIGINS=https://yourapp.hf.space,https://yourdomain.com
# Defaults to localhost-only when not set (safe for local dev).
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "MCP_ALLOWED_ORIGINS", "http://localhost:7860,http://127.0.0.1:7860"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)


def verify_token(request: Request):
    """Optional bearer token authentication."""
    if not MCP_AUTH_TOKEN:
        return  # auth disabled
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {MCP_AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing MCP_AUTH_TOKEN")


# SSE endpoint


@app.get("/sse", dependencies=[Depends(verify_token)])
async def sse_endpoint(request: Request):
    """
    SSE transport endpoint for MCP.
    The client sends JSON-RPC messages as query param `message` (URL-encoded JSON).
    This is the simplified SSE pattern compatible with most MCP clients.
    """

    async def event_generator():
        # Send endpoint event (required by MCP SSE spec)
        yield f"event: endpoint\ndata: /messages\n\n"
        # Keep-alive
        try:
            while True:
                if await request.is_disconnected():
                    break
                yield ": keep-alive\n\n"
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# JSON-RPC POST endpoint


@app.post("/messages", dependencies=[Depends(verify_token)])
async def messages_endpoint(request: Request):
    """Main MCP JSON-RPC 2.0 endpoint."""
    body = await request.json()

    # Handle batch requests
    if isinstance(body, list):
        responses = []
        for msg in body:
            resp = await handle_mcp_message(msg)
            if resp is not None:
                responses.append(resp)
        return responses

    resp = await handle_mcp_message(body)
    if resp is None:
        return {}
    return resp


# Health & info endpoints


@app.get("/health")
async def health():
    csv_ok = CSV_PATH.exists()
    return {
        "status": "ok",
        "csv_loaded": csv_ok,
        "csv_rows": len(_df_cache) if _df_cache is not None else 0,
        "mysql_driver": MYSQL_AVAILABLE,
        "postgres_driver": PG_AVAILABLE,
        "mysql_configured": bool(
            os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQL_HOST")
        ),
        "postgres_configured": bool(os.getenv("PG_PASSWORD") or os.getenv("PG_HOST")),
        "tools_count": len(MCP_TOOLS),
        "resources_count": len(MCP_RESOURCES),
    }


@app.get("/tools")
async def list_tools():
    return {"tools": MCP_TOOLS}


@app.get("/")
async def root():
    return {
        "server": "Pet Store SCM MCP Server",
        "version": "1.0.0",
        "transport": "SSE over HTTP",
        "endpoints": {
            "sse": "/sse",
            "messages": "/messages (POST)",
            "health": "/health",
            "tools": "/tools",
            "docs": "/docs",
        },
    }


# ── New tools (added for production reliability) ─────────────────────────────


def tool_get_transfer_recommendations(
    max_days_of_supply: float = 7.0,
    top_n: int = 15,
) -> str:
    """
    Stock transfer recommendations: identifies overstocked stores that can
    donate inventory to critically understocked stores for the same SKU,
    avoiding emergency purchase orders and reducing holding costs.
    Answers: "Should we transfer stock between stores?"
    """
    try:
        sdi_path = DATA_DIR / "store_daily_inventory.csv"
        if not sdi_path.exists():
            return (
                "store_daily_inventory.csv not found. Run: python data/generate_data.py"
            )
        sdi = pd.read_csv(sdi_path, parse_dates=["date"])
        latest = sdi["date"].max()
        sdi = sdi[sdi["date"] == latest].copy()

        critical = sdi[sdi["days_of_supply"] <= max_days_of_supply].copy()
        overstocked = sdi[sdi["days_of_supply"] >= 60].copy()

        if critical.empty:
            return f"✅ No stores have less than {max_days_of_supply} days of supply. No transfers needed."

        transfers = []
        for _, need_row in critical.iterrows():
            sku = need_row["sku_id"]
            donors = overstocked[overstocked["sku_id"] == sku]
            for _, donor in donors.iterrows():
                # BUG-003 fix: use actual daily demand not days_of_supply
                # days_of_supply = inventory/demand (a ratio), not demand itself
                avg_d = max(float(need_row.get("demand", 1)), 1.0)
                # Give receiving store 14 days of buffer stock
                need_units = max(0, int(14 * avg_d) - int(need_row["inventory"]))
                # Donor can give up to what keeps them above 60d buffer
                can_give = max(
                    0,
                    int(donor["inventory"])
                    - int(60 * max(float(donor.get("demand", 1)), 1.0)),
                )
                transfer_qty = min(need_units, can_give)
                if transfer_qty > 0:
                    transfers.append(
                        {
                            "sku_id": sku,
                            "name": str(need_row.get("name", ""))[:28],
                            "from_store": donor["store_id"],
                            "from_city": donor.get("city", ""),
                            "from_dos": float(donor["days_of_supply"]),
                            "to_store": need_row["store_id"],
                            "to_city": need_row.get("city", ""),
                            "to_dos": float(need_row["days_of_supply"]),
                            "transfer_qty": transfer_qty,
                            "urgency": "🔴 URGENT"
                            if float(need_row["days_of_supply"]) < 3
                            else "🟡 SOON",
                        }
                    )

        if not transfers:
            return (
                f"⚠️ {len(critical)} stores are critically low but no overstocked donors found "
                f"for the same SKUs. Recommend placing purchase orders via `generate_purchase_order`."
            )

        transfers_df = (
            pd.DataFrame(transfers)
            .sort_values(["urgency", "to_dos"], ascending=[True, True])
            .head(top_n)
        )

        header = (
            f"## Stock Transfer Recommendations (as of {latest.date()})\n\n"
            f"**{len(transfers_df)} transfer(s) identified** that avoid emergency POs "
            f"by moving surplus stock from overstocked to critical stores.\n\n"
        )
        tbl = (
            "| Urgency | SKU | Product | Qty to Transfer | From Store | From City | From DoS "
            "| To Store | To City | To DoS |\n"
            "|---------|-----|---------|----------------|------------|-----------|---------|"
            "----------|---------|--------|\n"
        )
        for _, r in transfers_df.iterrows():
            tbl += (
                f"| {r['urgency']} | {r['sku_id']} | {r['name']} | **{r['transfer_qty']:,} units** "
                f"| {r['from_store']} | {r['from_city']} | {r['from_dos']:.0f}d "
                f"| {r['to_store']} | {r['to_city']} | **{r['to_dos']:.1f}d** |\n"
            )
        rec = (
            "\n**How to action:** Coordinate with your logistics team to arrange "
            "inter-store transfers within 24–48 hours for 🔴 URGENT rows. "
            "This avoids rush shipping costs on emergency purchase orders.\n"
        )
        return header + tbl + rec
    except Exception as e:
        return f"TOOL_ERROR [get_transfer_recommendations]: {e}"


def tool_get_abc_xyz_analysis(
    category: str | None = None,
) -> str:
    """
    ABC-XYZ inventory classification:
      A/B/C = revenue contribution (A = top 70%, B = next 20%, C = bottom 10%)
      X/Y/Z = demand variability (X = stable CV<0.5, Y = variable 0.5-1.0, Z = erratic >1.0)
    Returns a Markdown table of all SKUs with their class and recommended
    stocking strategy (AX = always in stock, CZ = consider dropping, etc.)
    Answers: "Which are our most important SKUs?" / "What's our ABC analysis?"
    """
    try:
        dem = get_df()
        if dem.empty:
            return "No demand data available."

        latest_date = dem["date"].max()
        cutoff = latest_date - pd.Timedelta(days=90)
        recent = dem[dem["date"] >= cutoff].copy()

        if category:
            recent = recent[
                recent["category"].str.contains(category, case=False, na=False)
            ]

        # BUG-002 fix: pre-compute revenue column before groupby to avoid
        # lambda index misalignment when recent has been filtered/re-indexed
        recent["_revenue"] = recent["demand"] * recent["price_inr"]
        sku_stats = (
            recent.groupby(["sku_id", "name", "category"])
            .agg(
                total_revenue=("_revenue", "sum"),
                avg_demand=("demand", "mean"),
                std_demand=("demand", "std"),
            )
            .reset_index()
        )
        recent = recent.drop(columns=["_revenue"])
        sku_stats["cv"] = (
            sku_stats["std_demand"] / sku_stats["avg_demand"].replace(0, np.nan)
        ).fillna(0)

        # ABC classification
        total_rev = sku_stats["total_revenue"].sum()
        sku_stats = sku_stats.sort_values("total_revenue", ascending=False).reset_index(
            drop=True
        )
        sku_stats["cum_pct"] = (
            sku_stats["total_revenue"].cumsum() / max(total_rev, 1) * 100
        )
        sku_stats["abc"] = sku_stats["cum_pct"].apply(
            lambda x: "A" if x <= 70 else ("B" if x <= 90 else "C")
        )

        # XYZ classification
        sku_stats["xyz"] = sku_stats["cv"].apply(
            lambda x: "X" if x < 0.5 else ("Y" if x < 1.0 else "Z")
        )
        sku_stats["class"] = sku_stats["abc"] + sku_stats["xyz"]

        strategy_map = {
            "AX": "Always in stock — highest priority replenishment",
            "AY": "High stock — buffer for variability",
            "AZ": "Monitor closely — high revenue but erratic demand",
            "BX": "Standard replenishment cycle",
            "BY": "Regular stock with safety buffer",
            "BZ": "Review demand pattern — consider promotions",
            "CX": "Low stock, regular cycle",
            "CY": "Minimal stock, order on demand",
            "CZ": "Consider discontinuing or made-to-order",
        }

        scope = f" — {category}" if category else ""
        header = (
            f"## ABC-XYZ Inventory Classification{scope}\n\n"
            f"**{len(sku_stats)} SKUs** classified over last 90 days. "
            f"A = top 70% revenue | B = next 20% | C = bottom 10%  ·  "
            f"X = stable (CV<0.5) | Y = variable | Z = erratic (CV>1.0)\n\n"
        )
        tbl = (
            "| Class | SKU | Product | Category | Avg Daily Demand | CV | Revenue (₹) | Strategy |\n"
            "|-------|-----|---------|----------|------------------|----|------------|----------|\n"
        )
        for _, r in sku_stats.iterrows():
            tbl += (
                f"| **{r['class']}** | {r['sku_id']} | {str(r['name'])[:25]} | {r['category']} "
                f"| {r['avg_demand']:.1f} | {r['cv']:.2f} | ₹{r['total_revenue']:,.0f} "
                f"| {strategy_map.get(r['class'], '')} |\n"
            )
        return header + tbl
    except Exception as e:
        return f"TOOL_ERROR [get_abc_xyz_analysis]: {e}"


def tool_get_supplier_fill_rate_trend(
    supplier_name: str | None = None,
    months: int = 6,
) -> str:
    """
    Shows how each supplier's fill rate, OTD%, and defect rate have trended
    over the last N months. Identifies suppliers who are getting worse over
    time vs improving. Returns a Markdown table sorted by trend direction.
    Answers: "Which suppliers are getting worse?" / "How has supplier X's fill rate changed?"
    """
    try:
        sp = (
            get_supplier_perf()
        )  # BUG-001 fix: server.py uses get_supplier_perf() not get_supplier_perf_df()
        if sp.empty:
            return "No supplier performance data available."

        if supplier_name:
            sp = sp[
                sp["supplier_name"].str.contains(supplier_name, case=False, na=False)
            ]
            if sp.empty:
                return f"No data found for supplier matching '{supplier_name}'."

        # Get last N months
        sp["review_month"] = pd.to_datetime(
            sp["review_month"], format="%Y-%m", errors="coerce"
        )
        sp = sp.dropna(subset=["review_month"])
        cutoff = sp["review_month"].max() - pd.DateOffset(months=months)
        recent = sp[sp["review_month"] >= cutoff].copy()

        # Compute trend: slope of OTD over the period per supplier
        trends = []
        for sup, grp in recent.groupby("supplier_name"):
            grp = grp.sort_values("review_month")
            n = len(grp)
            otd_vals = grp["on_time_delivery_pct"].fillna(0).values
            fill_vals = (
                grp["fill_rate_pct"].fillna(0).values
                if "fill_rate_pct" in grp.columns
                else otd_vals
            )
            x = np.arange(n, dtype=float)
            otd_slope = float(np.polyfit(x, otd_vals, 1)[0]) if n >= 2 else 0.0
            fill_slope = float(np.polyfit(x, fill_vals, 1)[0]) if n >= 2 else 0.0
            trends.append(
                {
                    "supplier": sup,
                    "avg_otd": float(np.mean(otd_vals)),
                    "latest_otd": float(otd_vals[-1]),
                    "otd_trend_per_month": round(otd_slope, 2),
                    "avg_fill": float(np.mean(fill_vals)),
                    "latest_fill": float(fill_vals[-1]),
                    "fill_trend_per_month": round(fill_slope, 2),
                    "n_reviews": n,
                    "direction": (
                        "📈 Improving"
                        if otd_slope > 0.5
                        else "📉 Declining"
                        if otd_slope < -0.5
                        else "➡ Stable"
                    ),
                }
            )

        df_trends = pd.DataFrame(trends).sort_values("otd_trend_per_month")

        header = (
            f"## Supplier Fill Rate Trend — Last {months} Months\n\n"
            f"**{len(df_trends)} suppliers** · "
            f"📉 {df_trends['direction'].str.contains('Declining').sum()} declining · "
            f"📈 {df_trends['direction'].str.contains('Improving').sum()} improving\n\n"
        )
        tbl = (
            "| Trend | Supplier | Avg OTD% | Latest OTD% | OTD Δ/month "
            "| Avg Fill% | Latest Fill% | Fill Δ/month | Reviews |\n"
            "|-------|----------|---------|------------|----------|"
            "----------|-------------|-------------|--------|\n"
        )
        for _, r in df_trends.iterrows():
            otd_delta = (
                f"+{r['otd_trend_per_month']:.1f}"
                if r["otd_trend_per_month"] >= 0
                else f"{r['otd_trend_per_month']:.1f}"
            )
            fill_delta = (
                f"+{r['fill_trend_per_month']:.1f}"
                if r["fill_trend_per_month"] >= 0
                else f"{r['fill_trend_per_month']:.1f}"
            )
            tbl += (
                f"| {r['direction']} | {r['supplier'][:30]} "
                f"| {r['avg_otd']:.1f}% | {r['latest_otd']:.1f}% | **{otd_delta}%** "
                f"| {r['avg_fill']:.1f}% | {r['latest_fill']:.1f}% | **{fill_delta}%** "
                f"| {r['n_reviews']} |\n"
            )
        rec = (
            "\n**Action:** Suppliers marked 📉 Declining should be flagged for "
            "review. Use `get_supplier_negotiation_brief` for leverage analysis "
            "before the next contract renewal.\n"
        )
        return header + tbl + rec
    except Exception as e:
        return f"TOOL_ERROR [get_supplier_fill_rate_trend]: {e}"


def tool_get_basket_analysis(
    category: str | None = None,
    top_n: int = 15,
) -> str:
    """
    Market basket analysis: which products are most frequently bought together
    in the same transaction. Identifies cross-sell and bundle opportunities.
    Answers: "What do customers buy together?" / "Which products should we bundle?"
    """
    try:
        txn = get_transactions()
        if txn.empty:
            return "No transaction data available."

        if category:
            txn = txn[txn["category"].str.contains(category, case=False, na=False)]

        # BUG-023 fix: generated data has 1 SKU per txn_id (unique txn per row),
        # so groupby("txn_id") always gives single-item baskets.
        # Use store+date+customer_segment as a proxy basket: items bought at the
        # same store on the same day by customers of the same type are likely
        # purchased as part of the same shopping occasion.
        basket_key = (
            ["store_id", "date", "customer_segment"]
            if all(c in txn.columns for c in ["store_id", "date", "customer_segment"])
            else ["txn_id"]
        )
        baskets = txn.groupby(basket_key)["sku_id"].apply(list)
        # Only keep baskets with 2+ items
        baskets = baskets[baskets.apply(len) >= 2]

        if baskets.empty:
            return "No multi-item baskets found in the transaction data."

        # Count co-occurrence pairs
        pair_counts: dict = {}
        for items in baskets:
            unique_items = sorted(set(items))
            for i in range(len(unique_items)):
                for j in range(i + 1, len(unique_items)):
                    key = (unique_items[i], unique_items[j])
                    pair_counts[key] = pair_counts.get(key, 0) + 1

        if not pair_counts:
            return "No product pairs found."

        # Build results with support and product names
        sku_names = txn.drop_duplicates("sku_id").set_index("sku_id")["name"].to_dict()
        sku_cats = (
            txn.drop_duplicates("sku_id").set_index("sku_id")["category"].to_dict()
        )
        total_baskets = len(baskets)
        basket_method = (
            "store+date+segment proxy baskets"
            if basket_key != ["txn_id"]
            else "transaction baskets"
        )

        pairs_df = (
            pd.DataFrame(
                [
                    {"sku_a": k[0], "sku_b": k[1], "co_purchases": v}
                    for k, v in pair_counts.items()
                ]
            )
            .sort_values("co_purchases", ascending=False)
            .head(top_n)
        )
        pairs_df["support_pct"] = (
            pairs_df["co_purchases"] / total_baskets * 100
        ).round(1)

        scope = f" — {category}" if category else ""
        header = (
            f"## Basket Analysis — Frequently Bought Together{scope}\n\n"
            f"Analysed **{total_baskets:,} {basket_method}** from {len(txn):,} transaction rows. "
            f"Support % = % of baskets containing both products.\n\n"
        )
        tbl = (
            "| Rank | Product A | Category A | Product B | Category B "
            "| Co-Purchases | Support % | Bundle Opportunity |\n"
            "|------|-----------|-----------|-----------|-----------|"
            "-------------|----------|-------------------|\n"
        )
        for i, (_, r) in enumerate(pairs_df.iterrows(), 1):
            name_a = str(sku_names.get(r["sku_a"], r["sku_a"]))[:25]
            name_b = str(sku_names.get(r["sku_b"], r["sku_b"]))[:25]
            cat_a = sku_cats.get(r["sku_a"], "")
            cat_b = sku_cats.get(r["sku_b"], "")
            bundle = (
                "⭐ High"
                if r["support_pct"] >= 5
                else ("✅ Medium" if r["support_pct"] >= 2 else "Low")
            )
            tbl += (
                f"| {i} | {name_a} | {cat_a} | {name_b} | {cat_b} "
                f"| {int(r['co_purchases']):,} | **{r['support_pct']}%** | {bundle} |\n"
            )
        rec = (
            "\n**Use this for:** Shelf placement (put high-support pairs near each other), "
            "bundle promotions (e.g. 'Buy food, get 20% off matching treats'), "
            "and email recommendations ('Customers who bought X also bought Y').\n"
        )
        return header + tbl + rec
    except Exception as e:
        return f"TOOL_ERROR [get_basket_analysis]: {e}"


def tool_get_price_elasticity_analysis(
    category: str | None = None,
) -> str:
    """
    Estimates price elasticity of demand for each SKU: how much does demand
    change when the discount percentage changes? Uses promotion event data.
    Classifies SKUs as elastic (discount-sensitive), inelastic (price-insensitive),
    or negative-elastic (premium effect). Guides markdown and promotion decisions.
    Answers: "How sensitive is demand to price?" / "Which products respond to discounts?"
    """
    try:
        txn = get_transactions()
        promos = (
            get_promotions()
        )  # BUG-001 fix: server.py uses get_promotions() not get_promotions_df()
        dem = get_df()
        if txn.empty or promos.empty:
            return "Transaction or promotion data unavailable for elasticity analysis."

        if category:
            txn = txn[txn["category"].str.contains(category, case=False, na=False)]

        results = []
        for _, promo in promos.iterrows():
            start = pd.to_datetime(promo["start_date"])
            end = pd.to_datetime(promo["end_date"])
            discount = float(promo.get("discount_pct", 0))
            if discount <= 0:
                continue

            pre_window = dem[
                (dem["date"] >= start - pd.Timedelta(days=14)) & (dem["date"] < start)
            ]
            during_window = dem[(dem["date"] >= start) & (dem["date"] <= end)]

            if category:
                pre_window = pre_window[
                    pre_window["category"].str.contains(category, case=False, na=False)
                ]
                during_window = during_window[
                    during_window["category"].str.contains(
                        category, case=False, na=False
                    )
                ]

            pre_avg = float(pre_window.groupby("date")["demand"].sum().mean() or 0)
            during_avg = float(
                during_window.groupby("date")["demand"].sum().mean() or 0
            )

            if pre_avg > 0 and during_avg > 0:
                demand_change_pct = (during_avg - pre_avg) / pre_avg * 100
                elasticity = demand_change_pct / discount if discount > 0 else 0
                results.append(
                    {
                        "promo": str(promo.get("name", ""))[:30],
                        "category": str(
                            promo.get("target_category", category or "All")
                        ),
                        "discount_pct": discount,
                        "pre_avg_demand": round(pre_avg, 1),
                        "during_avg_demand": round(during_avg, 1),
                        "demand_change_pct": round(demand_change_pct, 1),
                        "elasticity": round(elasticity, 2),
                        "type": (
                            "🟢 Elastic"
                            if elasticity > 1.5
                            else "🟡 Moderate"
                            if elasticity > 0.5
                            else "🔵 Inelastic"
                            if elasticity >= 0
                            else "🔴 Negative"
                        ),
                    }
                )

        if not results:
            return "Not enough promotion data to compute elasticity."

        df_el = pd.DataFrame(results).sort_values("elasticity", ascending=False)
        avg_el = df_el["elasticity"].mean()

        scope = f" — {category}" if category else ""
        header = (
            f"## Price Elasticity Analysis{scope}\n\n"
            f"**{len(df_el)} promotion events** analysed. "
            f"Average elasticity: **{avg_el:.2f}** "
            f"(>1.5 = highly elastic, 0.5–1.5 = moderate, <0.5 = inelastic).\n\n"
        )
        tbl = (
            "| Type | Promotion | Category | Discount% | Pre-Promo Demand "
            "| During Demand | Demand Δ% | Elasticity |\n"
            "|------|-----------|----------|-----------|-----------------|"
            "--------------|----------|------------|\n"
        )
        for _, r in df_el.iterrows():
            tbl += (
                f"| {r['type']} | {r['promo']} | {r['category']} "
                f"| {r['discount_pct']:.0f}% | {r['pre_avg_demand']:.0f}/day "
                f"| {r['during_avg_demand']:.0f}/day "
                f"| **{r['demand_change_pct']:+.1f}%** | **{r['elasticity']:.2f}** |\n"
            )
        rec = (
            "\n**How to use this:**\n"
            "- 🟢 Elastic SKUs: discounts drive significant volume — use for clearance and traffic building\n"
            "- 🟡 Moderate: standard promotional response — good for seasonal drives\n"
            "- 🔵 Inelastic: customers buy regardless of price — don't over-discount these, protect margin\n"
            "- 🔴 Negative: discounting reduces demand (premium perception) — avoid discounting\n"
        )
        return header + tbl + rec
    except Exception as e:
        return f"TOOL_ERROR [get_price_elasticity_analysis]: {e}"


def tool_get_forecast_vs_actual(
    sku_id: str | None = None,
    days: int = 30,
) -> str:
    """
    Compares forecasted demand (from the prediction log) against actual demand
    for the same period. Computes MAPE, bias (over/under forecast), and flags
    SKUs with poor forecast accuracy so they can be retrained.
    Answers: "How accurate were our forecasts?" / "Which SKUs have the worst forecast error?"
    """
    try:
        log_path = Path(__file__).parent.parent / "logs" / "predictions.csv"
        if not log_path.exists():
            return (
                "No forecast log found. Go to the MLOps Monitor tab and run a "
                "forecast, then re-query this tool."
            )

        pred_df = pd.read_csv(log_path)
        if pred_df.empty:
            return "Forecast log is empty — run forecasts from the MLOps Monitor tab first."

        pred_df["logged_at"] = pd.to_datetime(pred_df["logged_at"])
        # One latest forecast per SKU
        pred_df = pred_df.sort_values("logged_at", ascending=False).drop_duplicates(
            "sku_id"
        )

        if sku_id:
            pred_df = pred_df[pred_df["sku_id"].str.upper() == sku_id.upper()]
            if pred_df.empty:
                return f"No forecast found for SKU '{sku_id}'."

        dem = get_df()
        latest_date = dem["date"].max()
        cutoff = latest_date - pd.Timedelta(days=days)
        actuals = (
            dem[dem["date"] >= cutoff]
            .groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "actual_avg_daily"})
        )

        merged = pred_df.merge(actuals, on="sku_id", how="inner")
        if merged.empty:
            return "Could not match forecasted SKUs to actual demand data."

        merged["forecast_avg_daily"] = merged["p50_daily"].fillna(
            merged.get("p50_total", pd.Series([0] * len(merged)))
            / merged.get("horizon_days", pd.Series([30] * len(merged)))
        )
        merged["error_pct"] = (
            (
                (merged["forecast_avg_daily"] - merged["actual_avg_daily"])
                / merged["actual_avg_daily"].replace(0, np.nan)
                * 100
            )
            .fillna(0)
            .round(1)
        )
        merged["mape"] = merged["error_pct"].abs().round(1)
        merged["bias"] = merged["error_pct"].apply(
            lambda x: (
                "📈 Over-forecast"
                if x > 10
                else ("📉 Under-forecast" if x < -10 else "✅ Accurate")
            )
        )
        merged["grade"] = merged["mape"].apply(
            lambda x: "A" if x < 10 else ("B" if x < 20 else ("C" if x < 35 else "D"))
        )
        merged = merged.sort_values("mape", ascending=False)

        avg_mape = float(merged["mape"].mean())
        scope = f" — {sku_id.upper()}" if sku_id else f" — Last {days} Days"

        header = (
            f"## Forecast vs Actual{scope}\n\n"
            f"**{len(merged)} SKU(s)** compared. "
            f"Average MAPE: **{avg_mape:.1f}%** "
            f"(A=<10%, B=<20%, C=<35%, D=≥35%)\n\n"
        )
        tbl = (
            "| Grade | SKU | Forecast (avg/day) | Actual (avg/day) "
            "| Error% | Bias | Logged At |\n"
            "|-------|-----|------------------|----------------|"
            "--------|------|----------|\n"
        )
        for _, r in merged.iterrows():
            tbl += (
                f"| **{r['grade']}** | {r['sku_id']} "
                f"| {r['forecast_avg_daily']:.1f} | {r['actual_avg_daily']:.1f} "
                f"| **{r['error_pct']:+.1f}%** | {r['bias']} "
                f"| {str(r['logged_at'])[:10]} |\n"
            )
        poor = merged[merged["grade"] == "D"]
        rec = (
            (
                f"\n**{len(poor)} SKU(s) with grade D** — recommend retraining. "
                "Go to MLOps Monitor → Fine-tune TFT to improve accuracy for these SKUs.\n"
            )
            if not poor.empty
            else "\n✅ All SKUs have acceptable forecast accuracy (grade A or B).\n"
        )
        return header + tbl + rec
    except Exception as e:
        return f"TOOL_ERROR [get_forecast_vs_actual]: {e}"


# ── Build the tool registry (must run after all tool functions are defined) ───
# Populates _TOOL_REGISTRY so that dispatch_tool() can use the fast registry
# path with TTL caching for all non-DB CSV tools.
_build_tool_registry()


# Direct callable interface (used by the agent's MCP client)


async def call_tool_direct(tool_name: str, arguments: dict) -> str:
    """
    Bypass HTTP — call a tool directly (used when MCP server runs in-process).
    The agent uses this when BYPASS_MCP_HTTP=true.
    """
    return await dispatch_tool(tool_name, arguments)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mcp_server.server:app",
        host="0.0.0.0",
        port=int(os.getenv("MCP_PORT", 8000)),
        reload=False,
        log_level="info",
    )
