"""
setup_databases.py
------------------
One-command script that sets up ALL databases — local AND cloud — automatically.

Run from the project root:
    python setup_databases.py

What it does:
    1. Reads ALL credentials from .env — no arguments needed
    2. Sets up Local MySQL        (MYSQL_HOST / MYSQL_PASSWORD)
    3. Sets up Local PostgreSQL   (PG_HOST / PG_PASSWORD)
    4. Sets up Cloud MySQL        (MYSQL_CLOUD_HOST / MYSQL_CLOUD_PASSWORD)
    5. Sets up Cloud PostgreSQL   (PG_CLOUD_HOST / PG_CLOUD_PASSWORD)

Any target whose HOST is blank in .env is skipped automatically.
Any target whose PASSWORD is blank is skipped with a warning.

Fill in .env and re-run — that is all you need to do.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")


# ── Simple console helpers (ASCII-only so Windows cp1252 doesn't crash) ───────
def ok(msg):
    print(f"  [OK]   {msg}")


def info(msg):
    print(f"  ...    {msg}")


def warn(msg):
    print(f"  [SKIP] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def banner(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


# ── Credential helpers ────────────────────────────────────────────────────────


def _local_mysql():
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": os.getenv("MYSQL_PORT", "3306"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "db": os.getenv("MYSQL_DB", "pet_store_scm"),
    }


def _local_pg():
    return {
        "host": os.getenv("PG_HOST", "localhost"),
        "port": os.getenv("PG_PORT", "5432"),
        "user": os.getenv("PG_USER", "postgres"),
        "password": os.getenv("PG_PASSWORD", ""),
        "db": os.getenv("PG_DB", "pet_store_scm"),
    }


def _cloud_mysql():
    host = os.getenv("MYSQL_CLOUD_HOST", "").strip()
    if not host:
        return None
    return {
        "host": host,
        "port": os.getenv("MYSQL_CLOUD_PORT", "3306"),
        "user": os.getenv("MYSQL_CLOUD_USER", "root"),
        "password": os.getenv("MYSQL_CLOUD_PASSWORD", ""),
        "db": os.getenv("MYSQL_CLOUD_DB", "railway"),
    }


def _cloud_pg():
    host = os.getenv("PG_CLOUD_HOST", "").strip()
    if not host:
        return None
    return {
        "host": host,
        "port": os.getenv("PG_CLOUD_PORT", "5432"),
        "user": os.getenv("PG_CLOUD_USER", "postgres"),
        "password": os.getenv("PG_CLOUD_PASSWORD", ""),
        "db": os.getenv("PG_CLOUD_DB", "railway"),
    }


# ── Core setup functions (delegate to original db/setup.py logic) ─────────────


def _apply_creds_to_env(creds: dict, is_mysql: bool) -> None:
    """Temporarily override the env vars that db/setup.py reads."""
    if is_mysql:
        os.environ["MYSQL_HOST"] = creds["host"]
        os.environ["MYSQL_PORT"] = str(creds["port"])
        os.environ["MYSQL_USER"] = creds["user"]
        os.environ["MYSQL_PASSWORD"] = creds["password"]
        os.environ["MYSQL_DB"] = creds["db"]
    else:
        os.environ["PG_HOST"] = creds["host"]
        os.environ["PG_PORT"] = str(creds["port"])
        os.environ["PG_USER"] = creds["user"]
        os.environ["PG_PASSWORD"] = creds["password"]
        os.environ["PG_DB"] = creds["db"]


def _run(creds: dict, is_mysql: bool, label: str, df) -> bool:
    """Override env vars, then call the original setup function. Returns True on success."""
    _apply_creds_to_env(creds, is_mysql)

    # Re-import setup functions AFTER overriding env so get_mysql_cfg / get_pg_cfg
    # pick up the new values (they read os.getenv at call time, not import time).
    try:
        if is_mysql:
            from db.setup import setup_mysql

            setup_mysql(df)
        else:
            from db.setup import setup_postgres

            setup_postgres(df)
        return True
    except SystemExit:
        # db/setup.py calls sys.exit(1) on connection failure
        return False
    except Exception as e:
        fail(str(e))
        return False


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("\nPet Store SCM - Full Database Setup")
    print("Reads all credentials from .env — no arguments needed.\n")

    # Load the CSV once — shared by all four targets.
    # Prefer the full HUFT dataset (has cost_inr, margin_pct, etc.).
    # Fall back to the legacy 11-column file if HUFT isn't generated yet.
    import importlib, pandas as pd

    huft_path = BASE_DIR / "data" / "huft_daily_demand.csv"
    legacy_path = BASE_DIR / "data" / "pet_store_supply_chain.csv"

    if huft_path.exists():
        info(f"Loading HUFT demand CSV ({huft_path.name})...")
        df = pd.read_csv(huft_path, parse_dates=["date"])
        # Alias price_inr → price_usd for legacy schema compatibility
        if "price_inr" in df.columns and "price_usd" not in df.columns:
            df["price_usd"] = df["price_inr"]
        ok(f"Loaded {len(df):,} rows from {huft_path.name}")
    elif legacy_path.exists():
        info(f"HUFT CSV not found — loading legacy CSV ({legacy_path.name})...")
        df = pd.read_csv(legacy_path, parse_dates=["date"])
        ok(f"Loaded {len(df):,} rows from {legacy_path.name}")
    else:
        info("No CSV found — generating synthetic data (takes ~30s)...")
        # Force the module to reload so it uses the correct BASE_DIR
        import db.setup as _s

        importlib.reload(_s)
        df = _s.generate_data()

    results: dict[str, str] = {}

    # ── 1. Local MySQL ────────────────────────────────────────────────────────
    banner("1 / 4   Local MySQL")
    creds = _local_mysql()
    print(f"  Target: {creds['user']}@{creds['host']}:{creds['port']}/{creds['db']}")
    if not creds["password"]:
        warn("MYSQL_PASSWORD is empty in .env — skipping.")
        results["Local MySQL"] = "skipped (no password in .env)"
    else:
        results["Local MySQL"] = "OK" if _run(creds, True, "local", df) else "FAILED"

    # ── 2. Local PostgreSQL ───────────────────────────────────────────────────
    banner("2 / 4   Local PostgreSQL")
    creds = _local_pg()
    print(f"  Target: {creds['user']}@{creds['host']}:{creds['port']}/{creds['db']}")
    if not creds["password"]:
        warn("PG_PASSWORD is empty in .env — skipping.")
        results["Local PostgreSQL"] = "skipped (no password in .env)"
    else:
        results["Local PostgreSQL"] = (
            "OK" if _run(creds, False, "local", df) else "FAILED"
        )

    # ── 3. Cloud MySQL ────────────────────────────────────────────────────────
    banner("3 / 4   Cloud MySQL  (MYSQL_CLOUD_* in .env)")
    creds = _cloud_mysql()
    if creds is None:
        warn("MYSQL_CLOUD_HOST is empty in .env — skipping cloud MySQL.")
        results["Cloud MySQL"] = "skipped (MYSQL_CLOUD_HOST not set)"
    elif not creds["password"]:
        warn("MYSQL_CLOUD_PASSWORD is empty in .env — skipping.")
        results["Cloud MySQL"] = "skipped (no password in .env)"
    else:
        print(
            f"  Target: {creds['user']}@{creds['host']}:{creds['port']}/{creds['db']}"
        )
        results["Cloud MySQL"] = "OK" if _run(creds, True, "cloud", df) else "FAILED"

    # ── 4. Cloud PostgreSQL ───────────────────────────────────────────────────
    banner("4 / 4   Cloud PostgreSQL  (PG_CLOUD_* in .env)")
    creds = _cloud_pg()
    if creds is None:
        warn("PG_CLOUD_HOST is empty in .env — skipping cloud PostgreSQL.")
        results["Cloud PostgreSQL"] = "skipped (PG_CLOUD_HOST not set)"
    elif not creds["password"]:
        warn("PG_CLOUD_PASSWORD is empty in .env — skipping.")
        results["Cloud PostgreSQL"] = "skipped (no password in .env)"
    else:
        print(
            f"  Target: {creds['user']}@{creds['host']}:{creds['port']}/{creds['db']}"
        )
        results["Cloud PostgreSQL"] = (
            "OK" if _run(creds, False, "cloud", df) else "FAILED"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Summary")
    for target, status in results.items():
        tag = (
            "[OK]  "
            if status == "OK"
            else ("[SKIP]" if "skipped" in status else "[FAIL]")
        )
        print(f"  {tag}  {target:<24}  {status}")

    print()
    any_failed = any("FAILED" in v for v in results.values())
    if any_failed:
        print("  Some targets FAILED — check the errors above.\n")
    else:
        print("  All done.  Run:  python gradio_app.py\n")


if __name__ == "__main__":
    main()
