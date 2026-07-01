"""Export the generated CSVs to a Parquet raw layer for the DuckDB mart build.

This is the "cheap, heavy storage" layer from the architecture: the full raw
data lives as compressed columnar Parquet (locally under data/parquet/, or in
object storage like S3 / Cloudflare R2 / a Hugging Face dataset in production),
and DuckDB reads it directly — no database server needed.

The Parquet file names match the dbt source table names in
dbt/models/staging/_sources.yml (products, stores, transactions, demand,
suppliers, store_inventory), so dbt-duckdb's `external_location` resolves them.

    python db/export_parquet.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "parquet"

# dbt source name -> source CSV.  Keys MUST match the `raw` source tables.
TABLES = {
    "products": "huft_products.csv",
    "stores": "huft_stores.csv",
    "transactions": "huft_sales_transactions.csv",
    "demand": "huft_daily_demand.csv",
    "suppliers": "huft_supplier_performance.csv",
    "store_inventory": "store_daily_inventory.csv",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for table, csv in TABLES.items():
        src = DATA / csv
        if not src.exists():
            raise FileNotFoundError(
                f"{src} not found — run `python data/generate_data.py` first."
            )
        df = pd.read_csv(src)
        dest = OUT / f"{table}.parquet"
        df.to_parquet(dest, index=False)
        print(f"  [OK] {table:16s} {len(df):>9,} rows -> {dest.relative_to(BASE)}")
    print(f"\nParquet raw layer written to {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
