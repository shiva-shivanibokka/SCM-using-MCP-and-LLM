"""
forecasting/data_loader.py
──────────────────────────
Loads training data for the TFT / CatBoost demand-forecasting model from:

  • MySQL      — daily_demand JOIN skus  (primary transactional source)
  • PostgreSQL — same query via asyncpg  (for companies using Postgres)
  • CSV        — local flat file         (development / demo fallback)

The training SQL is identical for both databases (standard SQL).
Only the driver and connection parameters differ.

Usage
-----
    from forecasting.data_loader import load_training_data

    df, description = load_training_data(
        source="mysql",
        mysql_creds={"host": "localhost", "port": 3306,
                     "user": "root", "password": "pw", "db": "pet_store_scm"},
    )
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ── SQL — works on both MySQL and PostgreSQL ─────────────────────────────────
# Uses COALESCE to handle both schema variants:
#   - migrate_huft.py schema: has price_inr, cost_inr, margin_pct
#   - setup.py legacy schema: has price_usd, no cost_inr / margin_pct
# The model uses price_usd internally so we alias price_inr → price_usd here.
TRAINING_SQL = """
SELECT
    record_date         AS date,
    sku_id,
    name,
    brand,
    brand_type,
    category,
    subcategory,
    pet_type,
    life_stage,
    supplier,
    demand,
    inventory,
    lead_time_days,
    COALESCE(price_inr, price_usd, 0)  AS price_inr,
    COALESCE(price_inr, price_usd, 0)  AS price_usd,
    COALESCE(cost_inr,  0)             AS cost_inr,
    COALESCE(margin_pct, 0)            AS margin_pct,
    is_cold_chain
FROM daily_demand
ORDER BY sku_id, record_date
"""

DataSource = Literal["mysql", "postgres", "csv"]

# ── Async database loaders ────────────────────────────────────────────────────


async def _fetch_mysql(creds: dict) -> pd.DataFrame:
    """Async: pull training data from MySQL via aiomysql."""
    try:
        import aiomysql
    except ImportError as exc:
        raise ImportError(
            "aiomysql is required for MySQL data loading. "
            "Install with: pip install aiomysql"
        ) from exc

    conn = await aiomysql.connect(
        host=creds.get("host", "localhost"),
        port=int(creds.get("port", 3306)),
        user=creds.get("user", "root"),
        password=creds.get("password", ""),
        db=creds.get("db", "pet_store_scm"),
        autocommit=True,
    )
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(TRAINING_SQL)
            rows = await cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # MySQL returns DECIMAL columns as Python decimal.Decimal objects.
    # Cast all numeric columns to float so numpy ufuncs (log1p, std, etc.) work.
    for col in [
        "demand",
        "inventory",
        "lead_time_days",
        "price_inr",
        "cost_inr",
        "margin_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


async def _fetch_postgres(creds: dict) -> pd.DataFrame:
    """Async: pull training data from PostgreSQL via asyncpg."""
    try:
        import asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required for PostgreSQL data loading. "
            "Install with: pip install asyncpg"
        ) from exc

    conn = await asyncpg.connect(
        host=creds.get("host", "localhost"),
        port=int(creds.get("port", 5432)),
        user=creds.get("user", "postgres"),
        password=creds.get("password", ""),
        database=creds.get("db", "pet_store_scm"),
    )
    try:
        rows = await conn.fetch(TRAINING_SQL)
    finally:
        await conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Synchronous wrappers ──────────────────────────────────────────────────────


def _run(coro):
    """Run an async coroutine from synchronous code.

    Handles three cases:
    1. Inside a running event loop (Gradio's async context) — offload to a
       fresh thread so we never call asyncio.run() inside an existing loop.
    2. No running loop — use asyncio.run() directly (Python 3.10+ safe).
    """
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        # A loop is already running (e.g. Gradio's event loop).
        # Run in a separate thread that has its own fresh event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly.
        return asyncio.run(coro)


def load_from_mysql(creds: dict) -> pd.DataFrame:
    """Load training data from MySQL (synchronous wrapper)."""
    logger.info("[DataLoader] Connecting to MySQL …")
    df = _run(_fetch_mysql(creds))
    logger.info(
        f"[DataLoader] MySQL: {len(df):,} rows | "
        f"{df['sku_id'].nunique()} SKUs | "
        f"date range {df['date'].min().date()} – {df['date'].max().date()}"
    )
    return df


def load_from_postgres(creds: dict) -> pd.DataFrame:
    """Load training data from PostgreSQL (synchronous wrapper)."""
    logger.info("[DataLoader] Connecting to PostgreSQL …")
    df = _run(_fetch_postgres(creds))
    logger.info(
        f"[DataLoader] PostgreSQL: {len(df):,} rows | "
        f"{df['sku_id'].nunique()} SKUs | "
        f"date range {df['date'].min().date()} – {df['date'].max().date()}"
    )
    return df


def load_from_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """Load training data from the local CSV (fallback / demo mode).

    Supports both the legacy pet_store_supply_chain.csv and the new
    huft_daily_demand.csv schema.  Column mapping applied:
      price_inr  → price_usd  (backward-compat alias)
    """
    if csv_path is None:
        # Prefer the HUFT dataset; fall back to legacy file
        huft_path = Path(__file__).parent.parent / "data" / "huft_daily_demand.csv"
        legacy_path = (
            Path(__file__).parent.parent / "data" / "pet_store_supply_chain.csv"
        )
        csv_path = huft_path if huft_path.exists() else legacy_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Run data/generate_data.py to create it."
        )
    logger.info(f"[DataLoader] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # ── HUFT schema normalisation ─────────────────────────────────────────
    # Rename price_inr → price_usd for backward compatibility with the model
    if "price_inr" in df.columns and "price_usd" not in df.columns:
        df = df.rename(columns={"price_inr": "price_usd"})

    # Ensure 'category' column is present (already in huft_daily_demand)
    if "category" not in df.columns:
        df["category"] = "Unknown"

    logger.info(
        f"[DataLoader] CSV: {len(df):,} rows | "
        f"{df['sku_id'].nunique()} SKUs | "
        f"date range {df['date'].min().date()} – {df['date'].max().date()}"
    )
    return df


# ── Main entry point ──────────────────────────────────────────────────────────


def load_training_data(
    source: DataSource = "csv",
    mysql_creds: dict | None = None,
    pg_creds: dict | None = None,
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load demand-forecasting training data from the chosen source.

    Parameters
    ----------
    source      : "mysql" | "postgres" | "csv"
    mysql_creds : dict  — host, port, user, password, db
    pg_creds    : dict  — host, port, user, password, db
    csv_path    : Path  — override default CSV location

    Returns
    -------
    (DataFrame, human-readable source description)

    The returned DataFrame always has columns:
        date, sku_id, demand, inventory, lead_time_days, price_usd, category
    """
    errors: list[str] = []

    # ── MySQL ──────────────────────────────────────────────────────────────
    if source == "mysql":
        if not mysql_creds:
            raise ValueError("mysql_creds required when source='mysql'.")
        try:
            df = load_from_mysql(mysql_creds)
            desc = (
                f"MySQL @ {mysql_creds.get('host', 'localhost')} | "
                f"{df['sku_id'].nunique()} SKUs | {len(df):,} rows"
            )
            return df, desc
        except Exception as exc:
            errors.append(f"MySQL failed: {exc}")
            logger.warning(f"[DataLoader] {errors[-1]} — trying CSV fallback")

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    if source == "postgres":
        if not pg_creds:
            raise ValueError("pg_creds required when source='postgres'.")
        try:
            df = load_from_postgres(pg_creds)
            desc = (
                f"PostgreSQL @ {pg_creds.get('host', 'localhost')} | "
                f"{df['sku_id'].nunique()} SKUs | {len(df):,} rows"
            )
            return df, desc
        except Exception as exc:
            errors.append(f"PostgreSQL failed: {exc}")
            logger.warning(f"[DataLoader] {errors[-1]} — trying CSV fallback")

    # ── CSV fallback ───────────────────────────────────────────────────────
    try:
        df = load_from_csv(csv_path)
        suffix = f" (fallback — {'; '.join(errors)})" if errors else ""
        desc = f"CSV{suffix} | {df['sku_id'].nunique()} SKUs | {len(df):,} rows"
        return df, desc
    except Exception as exc:
        raise RuntimeError(
            f"All data sources failed.\n" + "\n".join(errors) + f"\nCSV: {exc}"
        ) from exc
