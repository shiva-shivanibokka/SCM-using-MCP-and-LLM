"""Cached pandas loaders over the data/ CSVs."""
from __future__ import annotations

import functools
import pandas as pd

from .config import settings

_D = settings.DATA_DIR


@functools.lru_cache(maxsize=None)
def load_products() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_products.csv")


@functools.lru_cache(maxsize=None)
def load_stores() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_stores.csv")


@functools.lru_cache(maxsize=None)
def load_demand() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_daily_demand.csv", parse_dates=["date"])


@functools.lru_cache(maxsize=None)
def load_customers() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_customers.csv")


@functools.lru_cache(maxsize=None)
def load_transactions() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_sales_transactions.csv")


@functools.lru_cache(maxsize=None)
def load_suppliers() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_supplier_performance.csv")


@functools.lru_cache(maxsize=None)
def load_returns() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_returns.csv")


@functools.lru_cache(maxsize=None)
def load_promotions() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_promotions.csv")


def sku_history(sku_id: str) -> list[float]:
    df = load_demand()
    s = (df[df["sku_id"] == sku_id]
         .sort_values("date")
         .groupby("date")["demand"].sum())
    return [float(x) for x in s.tolist()]


def clear_cache() -> None:
    for fn in (load_products, load_stores, load_demand, load_customers,
               load_transactions, load_suppliers, load_returns, load_promotions):
        fn.cache_clear()
