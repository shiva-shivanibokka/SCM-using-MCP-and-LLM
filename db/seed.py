"""Seed Postgres DIRECTLY from generated data — no CSV middleman.

This is the primary way to populate the warehouse. It generates the synthetic
HUFT dataset in memory and writes each table straight into Postgres. The CSV
files generate_data.py also writes are kept ONLY as an offline fallback
(see backend/data_access.py, which reads Postgres first and falls back to CSV).

Run once after setting DATABASE_URL in .env:

    python db/seed.py

Idempotent: re-running replaces every table (generation is seeded, so the data
is identical each time).
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BASE / "backend" / ".env")
load_dotenv(BASE / ".env")

from backend.db import get_engine  # noqa: E402

# Tables written straight from the in-memory frames. The 1.2M-row daily
# inventory history is collapsed to a compact latest snapshot below.
DIRECT = ["products", "stores", "demand", "customers", "promotions",
          "transactions", "returns", "suppliers", "cold_chain"]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_txn_store ON transactions (store_id)",
    "CREATE INDEX IF NOT EXISTS idx_txn_sku ON transactions (sku_id)",
    "CREATE INDEX IF NOT EXISTS idx_demand_sku ON demand (sku_id)",
    "CREATE INDEX IF NOT EXISTS idx_sinv_store ON store_inventory (store_id)",
    "CREATE INDEX IF NOT EXISTS idx_sinv_sku ON store_inventory (sku_id)",
]


def _write(df, table, engine) -> None:
    df.to_sql(table, engine, if_exists="replace", index=False,
              method="multi", chunksize=1000)
    print(f"  [OK] {table:20s} {len(df):>9,} rows")


def main() -> None:
    engine = get_engine()
    if engine is None:
        print("ERROR: DATABASE_URL not set. Add it to .env first:")
        print("  DATABASE_URL=postgresql://user:pass@host.neon.tech/dbname")
        sys.exit(1)

    from data.generate_data import generate

    print("Generating synthetic HUFT dataset in memory…")
    frames = generate()

    # dbt builds views/marts in the `analytics` schema that depend on these raw
    # tables. Postgres won't let us replace a raw table while a view depends on
    # it, so drop the derived schema first — dbt rebuilds it in the next step
    # (`python db/run_dbt.py build`). This keeps re-seeding idempotent.
    with engine.begin() as conn:
        from sqlalchemy import text

        conn.execute(text("DROP SCHEMA IF EXISTS analytics CASCADE"))
    print("Cleared stale analytics schema (dbt will rebuild it).")

    print("\nWriting tables directly to Postgres (raw layer)…")
    for table in DIRECT:
        _write(frames[table], table, engine)

    # Compact store_inventory: latest snapshot per (store, SKU).
    sdi = frames["store_daily_inventory"]
    latest = (sdi.sort_values("date")
              .groupby(["store_id", "sku_id"]).tail(1)
              .reset_index(drop=True))
    _write(latest, "store_inventory", engine)

    with engine.begin() as conn:
        from sqlalchemy import text
        for stmt in INDEXES:
            conn.execute(text(stmt))
    print("  [OK] indexes created")
    print("\nDone. Postgres seeded directly from generated data — no CSV load step.")
    print("Next: run the dbt transformations →  python db/run_dbt.py build")


if __name__ == "__main__":
    main()
