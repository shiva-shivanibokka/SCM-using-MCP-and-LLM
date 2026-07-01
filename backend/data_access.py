"""Data access layer.

Reads from PostgreSQL (Neon) when DATABASE_URL is set; otherwise falls back to
the bundled CSVs. Either way the rest of the app sees identical DataFrames, so
routes and forecasting don't care where the data lives.
"""
from __future__ import annotations

import functools
import logging

import pandas as pd

from .config import settings
from .db import get_engine

logger = logging.getLogger(__name__)
_D = settings.DATA_DIR


def _load(table: str, csv: str, parse_dates=None) -> pd.DataFrame:
    """Load a table from Postgres, falling back to the CSV on any failure."""
    engine = get_engine()
    if engine is not None:
        try:
            return pd.read_sql_table(table, engine, parse_dates=parse_dates)
        except Exception as e:  # table missing, network blip, etc.
            logger.warning("DB read failed for '%s' (%s) — using CSV", table, e)
    return pd.read_csv(_D / csv, parse_dates=parse_dates) if parse_dates \
        else pd.read_csv(_D / csv)


@functools.lru_cache(maxsize=None)
def load_products() -> pd.DataFrame:
    return _load("products", "huft_products.csv")


@functools.lru_cache(maxsize=None)
def load_stores() -> pd.DataFrame:
    return _load("stores", "huft_stores.csv")


@functools.lru_cache(maxsize=None)
def load_demand() -> pd.DataFrame:
    return _load("demand", "huft_daily_demand.csv", parse_dates=["date"])


@functools.lru_cache(maxsize=None)
def load_customers() -> pd.DataFrame:
    return _load("customers", "huft_customers.csv")


@functools.lru_cache(maxsize=None)
def load_transactions() -> pd.DataFrame:
    return _load("transactions", "huft_sales_transactions.csv", parse_dates=["date"])


@functools.lru_cache(maxsize=None)
def load_suppliers() -> pd.DataFrame:
    return _load("suppliers", "huft_supplier_performance.csv")


@functools.lru_cache(maxsize=None)
def load_returns() -> pd.DataFrame:
    return _load("returns", "huft_returns.csv")


@functools.lru_cache(maxsize=None)
def load_promotions() -> pd.DataFrame:
    return _load("promotions", "huft_promotions.csv")


@functools.lru_cache(maxsize=None)
def load_store_inventory() -> pd.DataFrame:
    # In Postgres this is the compact latest-snapshot table; from CSV it's the
    # full daily history. Callers take the latest row per (store, SKU) either way.
    return _load("store_inventory", "store_daily_inventory.csv", parse_dates=["date"])


def load_mart(name: str):
    """Read a dbt mart, preferring Postgres (`analytics` schema) and falling back
    to a locally-built DuckDB file (data/petopia.duckdb, from db/build_marts.py).
    Returns None if the mart isn't built anywhere so callers can fall back to raw
    compute. The duckdb import is optional — absent on the deployed backend, the
    fallback is simply skipped."""
    engine = get_engine()
    if engine is not None:
        try:
            return pd.read_sql_table(name, engine, schema="analytics")
        except Exception as e:
            logger.info("dbt mart '%s' not in Postgres (%s) — trying DuckDB", name, e)

    duck = _D / "petopia.duckdb"
    if duck.exists():
        try:
            import duckdb  # optional; only present where marts are built locally

            con = duckdb.connect(str(duck), read_only=True)
            try:
                return con.execute(f'select * from analytics."{name}"').df()
            finally:
                con.close()
        except Exception as e:
            logger.info("dbt mart '%s' not in DuckDB (%s) — using raw compute", name, e)
    return None


def sku_history(sku_id: str) -> list[float]:
    df = load_demand()
    s = (df[df["sku_id"] == sku_id]
         .sort_values("date")
         .groupby("date")["demand"].sum())
    return [float(x) for x in s.tolist()]


def clear_cache() -> None:
    for fn in (load_products, load_stores, load_demand, load_customers,
               load_transactions, load_suppliers, load_returns, load_promotions,
               load_store_inventory):
        fn.cache_clear()
