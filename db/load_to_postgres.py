"""ETL: load the generated CSVs into PostgreSQL (Neon).

Postgres is the system of record. This script is idempotent — re-running it
replaces each table. Run once after creating your Neon project:

    # set DATABASE_URL in backend/.env first (postgresql://user:pass@host/db)
    python db/load_to_postgres.py

Design note: the per-store daily inventory history is 1.2M rows. The dashboards
only ever use the *latest* snapshot per (store, SKU), so we load just that
(~15k rows) as `store_inventory` — keeping us well inside Neon's free tier.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(BASE / "backend" / ".env")
load_dotenv(BASE / ".env")

from backend.db import get_engine  # noqa: E402

DATA = BASE / "data"

# table name -> (csv file, date columns)
TABLES = {
    "products": ("huft_products.csv", None),
    "stores": ("huft_stores.csv", None),
    "customers": ("huft_customers.csv", None),
    "transactions": ("huft_sales_transactions.csv", ["date"]),
    "suppliers": ("huft_supplier_performance.csv", None),
    "returns": ("huft_returns.csv", ["return_date"]),
    "demand": ("huft_daily_demand.csv", ["date"]),
    "promotions": ("huft_promotions.csv", None),
}

# Indexes that speed up the dashboard / agent queries.
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_txn_store ON transactions (store_id)",
    "CREATE INDEX IF NOT EXISTS idx_txn_sku ON transactions (sku_id)",
    "CREATE INDEX IF NOT EXISTS idx_demand_sku ON demand (sku_id)",
    "CREATE INDEX IF NOT EXISTS idx_sinv_store ON store_inventory (store_id)",
    "CREATE INDEX IF NOT EXISTS idx_sinv_sku ON store_inventory (sku_id)",
]


def _write(df: pd.DataFrame, table: str, engine) -> None:
    df.to_sql(table, engine, if_exists="replace", index=False,
              method="multi", chunksize=1000)
    print(f"  [OK] {table:16s} {len(df):>8,} rows")


def main() -> None:
    engine = get_engine()
    if engine is None:
        print("ERROR: DATABASE_URL not set. Add it to backend/.env first:")
        print("  DATABASE_URL=postgresql://user:pass@host.neon.tech/dbname")
        sys.exit(1)

    print("Loading CSVs into PostgreSQL…")
    for table, (csv, dates) in TABLES.items():
        path = DATA / csv
        if not path.exists():
            print(f"  [skip] {table}: {csv} missing — run data/generate_data.py first")
            continue
        df = pd.read_csv(path, parse_dates=dates) if dates else pd.read_csv(path)
        _write(df, table, engine)

    # Compact store-inventory: latest snapshot per (store, SKU) only.
    sdi_path = DATA / "store_daily_inventory.csv"
    if sdi_path.exists():
        sdi = pd.read_csv(sdi_path, parse_dates=["date"])
        latest = (sdi.sort_values("date")
                  .groupby(["store_id", "sku_id"]).tail(1)
                  .reset_index(drop=True))
        _write(latest, "store_inventory", engine)

    with engine.begin() as conn:
        from sqlalchemy import text
        for stmt in INDEXES:
            conn.execute(text(stmt))
    print("  [OK] indexes created")
    print("\nDone. The backend will now read from PostgreSQL.")


if __name__ == "__main__":
    main()
