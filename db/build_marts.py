"""Build the analytics marts locally with dbt + DuckDB — no Postgres required.

This is the free-tier path: it exports the CSVs to Parquet (the raw layer), then
runs dbt against the DuckDB target so the same mart models in dbt/models/marts/
materialise into a local DuckDB file (data/petopia.duckdb). The backend reads
those marts via backend/data_access.load_mart() when no Postgres is configured.

    python db/build_marts.py            # export parquet + build recommender marts
    python db/build_marts.py --all      # build every mart on DuckDB

In production the very same models build on Postgres via `python db/run_dbt.py
build` — swap the engine, keep the code.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling db/ modules

# The recommender/inventory marts this project adds. `+model` also builds the
# staging models they depend on. Existing Postgres-oriented marts (store_kpis
# etc.) are skipped on DuckDB by default since the app reads them from Postgres.
DEFAULT_SELECT = [
    "+co_purchase_pairs",
    "+customer_product_history",
    "+sku_days_of_supply",
]


def main() -> None:
    # 1. Ensure the Parquet raw layer exists.
    from export_parquet import main as export_parquet  # noqa: E402

    print("Exporting CSVs to Parquet raw layer…")
    export_parquet()

    # 2. Point dbt-duckdb at the raw Parquet dir and the output DuckDB file.
    import os

    raw_dir = (BASE / "data" / "parquet").as_posix()
    duckdb_path = (BASE / "data" / "petopia.duckdb").as_posix()
    os.environ["RAW_DIR"] = raw_dir
    os.environ["DUCKDB_PATH"] = duckdb_path

    project = str(BASE / "dbt")
    build_all = "--all" in sys.argv[1:]
    select = [] if build_all else ["--select", *DEFAULT_SELECT]

    # Resolve the dbt executable that lives next to the running Python (venv),
    # falling back to bare "dbt" on PATH.
    scripts = Path(sys.executable).parent
    dbt_exe = next(
        (str(scripts / n) for n in ("dbt.exe", "dbt") if (scripts / n).exists()),
        "dbt",
    )

    # `dbt run` (models only) — the DuckDB target is the demo/serving build; the
    # relationship/data tests run on the Postgres target via run_dbt.py.
    cmd = [dbt_exe, "run", "--target", "duck",
           "--project-dir", project, "--profiles-dir", project, *select]
    print("\nRunning:", " ".join(cmd))
    print(f"  RAW_DIR={raw_dir}\n  DUCKDB_PATH={duckdb_path}\n")
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"\nMarts built into {duckdb_path} (schema: analytics).")
        print("The backend auto-reads them via load_mart() when DATABASE_URL is unset.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
