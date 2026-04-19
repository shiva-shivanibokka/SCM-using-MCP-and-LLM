"""
Pet Store SCM -- One-command database setup script.

What this script does, in order:
  1. Reads your DB credentials from the .env file
  2. Generates the synthetic CSV data (87,660 rows) if not already present
  3. Creates the MySQL database and all tables
  4. Seeds MySQL: skus, suppliers, daily_demand (all 87,660 rows), reorder_events,
     supplier_performance
  5. Creates the PostgreSQL database and all tables
  6. Seeds PostgreSQL with sample forecasts, alerts, and monthly KPIs so the
     agent has data to query immediately for the demo

Run from the project root:
    python db/setup.py

Optional flags:
    python db/setup.py --mysql-only
    python db/setup.py --postgres-only
    python db/setup.py --skip-data      (skip CSV generation, use existing CSV)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")


def get_mysql_cfg() -> dict:
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "db": os.getenv("MYSQL_DB", "pet_store_scm"),
    }


def get_pg_cfg() -> dict:
    return {
        "host": os.getenv("PG_HOST", "localhost"),
        "port": int(os.getenv("PG_PORT", 5432)),
        "user": os.getenv("PG_USER", "postgres"),
        "password": os.getenv("PG_PASSWORD", ""),
        "db": os.getenv("PG_DB", "pet_store_scm"),
    }


def step(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def info(msg: str) -> None:
    print(f"  ...   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def generate_data() -> "pd.DataFrame":
    step("Step 1 of 4 -- Generating synthetic CSV data")
    csv_path = BASE_DIR / "data" / "pet_store_supply_chain.csv"
    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path, parse_dates=["date"])
        ok(f"CSV already exists ({len(df):,} rows) -- skipping generation")
        return df

    info("Generating 87,660 rows of synthetic pet store demand data...")
    from data.generate_data import generate

    df = generate()
    ok(f"CSV generated: {csv_path} ({len(df):,} rows)")
    return df


def setup_mysql(df: "pd.DataFrame") -> None:
    step("Step 2 of 4 -- Setting up MySQL")
    try:
        import pymysql
    except ImportError:
        try:
            import MySQLdb as pymysql
        except ImportError:
            print("\n  MySQL driver not found.")
            print("  Install one with:  pip install pymysql")
            print("  Then re-run this script.\n")
            sys.exit(1)

    cfg = get_mysql_cfg()
    db_name = cfg.pop("db")

    info(f"Connecting to MySQL at {cfg['host']}:{cfg['port']} as '{cfg['user']}'...")
    try:
        conn = pymysql.connect(**cfg, charset="utf8mb4")
    except Exception as e:
        fail(f"Could not connect to MySQL: {e}")
        print("\n  Make sure:")
        print(
            "  1. MySQL is running (check Windows Services or run 'mysqladmin status')"
        )
        print("  2. MYSQL_USER and MYSQL_PASSWORD in your .env are correct")
        print("  3. The user has permission to create databases\n")
        sys.exit(1)

    cur = conn.cursor()

    info(f"Creating database '{db_name}' if it does not exist...")
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cur.execute(f"USE `{db_name}`")
    ok(f"Database '{db_name}' ready")

    info("Creating tables...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS skus (
            sku_id          VARCHAR(20)    PRIMARY KEY,
            name            VARCHAR(255)   NOT NULL,
            category        VARCHAR(100)   NOT NULL,
            subcategory     VARCHAR(100)   NOT NULL,
            supplier        VARCHAR(150)   NOT NULL,
            region          VARCHAR(100)   NOT NULL,
            lead_time_days  TINYINT UNSIGNED NOT NULL,
            price_usd       DECIMAL(10,2)  NOT NULL,
            base_demand     SMALLINT UNSIGNED NOT NULL,
            is_active       BOOLEAN        NOT NULL DEFAULT TRUE,
            created_at      TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_category (category),
            INDEX idx_supplier (supplier),
            INDEX idx_region   (region)
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_name        VARCHAR(150)  PRIMARY KEY,
            contact_email        VARCHAR(255),
            contact_phone        VARCHAR(50),
            country              VARCHAR(100),
            on_time_delivery_pct DECIMAL(5,2),
            quality_rating       DECIMAL(3,1),
            min_order_qty        SMALLINT UNSIGNED,
            emergency_capable    BOOLEAN DEFAULT FALSE,
            notes                TEXT,
            created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_demand (
            id              BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
            record_date     DATE              NOT NULL,
            sku_id          VARCHAR(20)       NOT NULL,
            demand          MEDIUMINT UNSIGNED NOT NULL DEFAULT 0,
            inventory       MEDIUMINT UNSIGNED NOT NULL DEFAULT 0,
            lead_time_days  TINYINT UNSIGNED  NOT NULL,
            price_usd       DECIMAL(10,2)     NOT NULL,
            price_inr       DECIMAL(10,2),
            cost_inr        DECIMAL(10,2),
            margin_pct      DECIMAL(5,2),
            UNIQUE KEY uq_date_sku (record_date, sku_id),
            INDEX idx_sku  (sku_id),
            INDEX idx_date (record_date),
            FOREIGN KEY (sku_id) REFERENCES skus(sku_id) ON UPDATE CASCADE
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reorder_events (
            id               BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
            event_date       DATE              NOT NULL,
            sku_id           VARCHAR(20)       NOT NULL,
            trigger_reason   ENUM('LOW_STOCK','SCHEDULED','EMERGENCY','FORECAST') NOT NULL,
            ordered_qty      MEDIUMINT UNSIGNED NOT NULL,
            expected_arrival DATE,
            actual_arrival   DATE,
            unit_cost        DECIMAL(10,2),
            status           ENUM('PENDING','IN_TRANSIT','RECEIVED','CANCELLED') NOT NULL DEFAULT 'PENDING',
            created_at       TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_event (event_date, sku_id, trigger_reason),
            INDEX idx_sku_date (sku_id, event_date),
            FOREIGN KEY (sku_id) REFERENCES skus(sku_id) ON UPDATE CASCADE
        ) ENGINE=InnoDB
    """)

    # BUG-19 fix: use review_month CHAR(7) to match migrate_huft.py schema.
    # setup.py previously used review_date DATE — but migrate_huft.py uses
    # review_month CHAR(7) and huft_supplier_performance.csv has review_month.
    # Running both scripts on the same DB would leave the old schema in place,
    # causing migrate_huft.py inserts to fail with "Unknown column review_month".
    cur.execute("""
        CREATE TABLE IF NOT EXISTS supplier_performance (
            id              BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
            supplier_name   VARCHAR(150)      NOT NULL,
            review_month    CHAR(7)           NOT NULL,
            on_time_pct     DECIMAL(5,2),
            defect_rate_pct DECIMAL(5,2),
            fill_rate_pct   DECIMAL(5,2),
            notes           TEXT,
            UNIQUE KEY uq_supplier_month (supplier_name, review_month),
            FOREIGN KEY (supplier_name) REFERENCES suppliers(supplier_name) ON UPDATE CASCADE
        ) ENGINE=InnoDB
    """)
    conn.commit()
    ok("All MySQL tables created")

    info("Seeding suppliers...")
    suppliers = [
        (
            "PawsSupply Co",
            "orders@pawssupply.com",
            "United States",
            96.20,
            4.7,
            50,
            True,
            "Primary distributor. Same-day emergency orders up to 500 units.",
        ),
        (
            "NutriPet Inc",
            "supply@nutripetin.com",
            "United States",
            92.50,
            4.5,
            100,
            False,
            "Specialty nutrition supplier. MOQ 100 units.",
        ),
        (
            "GlobalPet Dist",
            "intl@globalpetdist.com",
            "Mexico",
            88.10,
            4.1,
            200,
            False,
            "Latin America distributor. Customs delays possible.",
        ),
        (
            "TreatWorld LLC",
            "b2b@treatworld.com",
            "China",
            84.30,
            3.9,
            500,
            False,
            "Asia Pacific treats supplier. Quality incident Q4 2023, resolved Q1 2024.",
        ),
        (
            "HealthPet Labs",
            "orders@healthpetlabs.com",
            "United States",
            94.80,
            4.6,
            50,
            True,
            "Pharmaceutical-grade supplements. FDA registered facility.",
        ),
        (
            "VetPharma Supply",
            "vet@vetpharma.eu",
            "Germany",
            91.20,
            4.8,
            30,
            False,
            "EU-based vet pharma. Highest quality rating.",
        ),
        (
            "PetEssentials",
            "wholesale@petessentials.com",
            "United States",
            97.10,
            4.4,
            100,
            True,
            "Best on-time delivery. General accessories and consumables.",
        ),
        (
            "ToyPet Factory",
            "orders@toypetfactory.com",
            "China",
            80.50,
            3.7,
            1000,
            False,
            "Budget toy supplier. High MOQ. Lead time variance +-5 days.",
        ),
        (
            "RawPet Foods",
            "cold@rawpetfoods.com",
            "United States",
            95.00,
            4.9,
            20,
            True,
            "Premium raw/freeze-dried specialist. Cold chain logistics.",
        ),
    ]
    cur.executemany(
        """INSERT IGNORE INTO suppliers
           (supplier_name, contact_email, country, on_time_delivery_pct,
            quality_rating, min_order_qty, emergency_capable, notes)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        suppliers,
    )
    conn.commit()
    ok(f"Seeded {len(suppliers)} suppliers")

    info("Seeding SKU master from CSV...")
    sku_rows = (
        df[
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
        .copy()
    )
    sku_rows["base_demand"] = (
        df.groupby("sku_id")["demand"]
        .mean()
        .round()
        .astype(int)
        .reindex(sku_rows["sku_id"])
        .values
    )
    sku_data = [
        (
            r.sku_id,
            r.name,
            r.category,
            r.subcategory,
            r.supplier,
            r.region,
            int(r.lead_time_days),
            float(r.price_usd),
            int(r.base_demand),
        )
        for r in sku_rows.itertuples()
    ]
    cur.executemany(
        """INSERT IGNORE INTO skus
           (sku_id, name, category, subcategory, supplier, region,
            lead_time_days, price_usd, base_demand)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        sku_data,
    )
    conn.commit()
    ok(f"Seeded {len(sku_data)} SKUs")

    info("Checking existing daily_demand rows...")
    cur.execute("SELECT COUNT(*) FROM daily_demand")
    existing = cur.fetchone()[0]
    if existing > 0:
        ok(f"daily_demand already has {existing:,} rows -- skipping bulk insert")
    else:
        info(
            f"Inserting {len(df):,} rows into daily_demand (this takes ~30-60 seconds)..."
        )
        batch_size = 5000
        # BUG-046 fix: include price_inr, cost_inr, margin_pct so financial tools
        # don't silently produce ₹0 values for all setup.py-seeded databases.
        rows = [
            (
                str(r.date.date()) if hasattr(r.date, "date") else str(r.date),
                r.sku_id,
                int(r.demand),
                int(r.inventory),
                int(r.lead_time_days),
                float(r.price_usd),
                float(getattr(r, "price_inr", r.price_usd)),
                float(getattr(r, "cost_inr", 0.0)),
                float(getattr(r, "margin_pct", 0.0)),
            )
            for r in df.itertuples()
        ]
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            cur.executemany(
                """INSERT IGNORE INTO daily_demand
                   (record_date, sku_id, demand, inventory, lead_time_days,
                    price_usd, price_inr, cost_inr, margin_pct)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                batch,
            )
            conn.commit()
            pct = min(100, int((i + batch_size) / len(rows) * 100))
            print(
                f"\r  ...   Loading daily_demand: {pct}% ({i + batch_size:,}/{len(rows):,} rows)",
                end="",
                flush=True,
            )
        print()
        ok(f"Inserted {len(rows):,} rows into daily_demand")

    info("Seeding sample reorder events...")
    import random

    random.seed(42)
    sku_ids = df["sku_id"].unique().tolist()
    reorder_rows = []
    base = date(2024, 1, 1)
    reasons = ["LOW_STOCK", "SCHEDULED", "EMERGENCY", "FORECAST"]
    statuses = ["RECEIVED", "RECEIVED", "RECEIVED", "IN_TRANSIT", "PENDING"]
    for i in range(120):
        ev_date = base + timedelta(days=random.randint(0, 364))
        sku = random.choice(sku_ids)
        reason = random.choice(reasons)
        qty = random.randint(200, 3000)
        arrival = ev_date + timedelta(days=random.randint(5, 21))
        actual = arrival + timedelta(days=random.randint(-2, 5))
        status = random.choice(statuses)
        cost = round(random.uniform(8.0, 60.0), 2)
        reorder_rows.append(
            (str(ev_date), sku, reason, qty, str(arrival), str(actual), cost, status)
        )
    # BUG-030 fix: use INSERT IGNORE so re-running setup doesn't duplicate events
    cur.executemany(
        """INSERT IGNORE INTO reorder_events
           (event_date, sku_id, trigger_reason, ordered_qty, expected_arrival,
            actual_arrival, unit_cost, status)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        reorder_rows,
    )
    conn.commit()
    ok(f"Seeded {len(reorder_rows)} reorder events")

    info("Seeding supplier performance reviews (quarterly 2022-2024)...")
    perf_rows = []
    for supplier, _, _, otd, qual, _, _, _ in suppliers:
        for yr in [2022, 2023, 2024]:
            for qtr in [("03", 31), ("06", 30), ("09", 30), ("12", 31)]:
                month, day = qtr
                # BUG-19 fix: use YYYY-MM format (review_month) not YYYY-MM-DD (review_date)
                rev_month = f"{yr}-{month}"
                jitter = lambda x: round(
                    min(100, max(50, x + random.uniform(-3, 3))), 2
                )
                perf_rows.append(
                    (
                        supplier,
                        rev_month,
                        jitter(float(otd)),
                        round(random.uniform(0.2, 2.5), 2),
                        round(random.uniform(94, 99.5), 2),
                        f"Quarterly review Q{['', '1', '2', '3', '4'][int(month) // 3]} {yr}",
                    )
                )
    cur.executemany(
        """INSERT IGNORE INTO supplier_performance
           (supplier_name, review_month, on_time_pct, defect_rate_pct, fill_rate_pct, notes)
           VALUES (%s,%s,%s,%s,%s,%s)""",
        perf_rows,
    )
    conn.commit()
    ok(f"Seeded {len(perf_rows)} supplier performance rows")

    cur.close()
    conn.close()
    ok("MySQL setup complete")


def setup_postgres(df: "pd.DataFrame") -> None:
    step("Step 3 of 4 -- Setting up PostgreSQL")
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("\n  PostgreSQL driver not found.")
        print("  Install with:  pip install psycopg2-binary")
        print("  Then re-run this script.\n")
        sys.exit(1)

    cfg = get_pg_cfg()
    db_name = cfg["db"]

    info(
        f"Connecting to PostgreSQL at {cfg['host']}:{cfg['port']} as '{cfg['user']}'..."
    )

    def connect(dbname: str):
        try:
            return psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["user"],
                password=cfg["password"],
                dbname=dbname,
            )
        except Exception as e:
            fail(f"Could not connect to PostgreSQL: {e}")
            print("\n  Make sure:")
            print("  1. PostgreSQL is running")
            print("  2. PG_USER and PG_PASSWORD in your .env are correct")
            print("  3. The user has permission to create databases\n")
            sys.exit(1)

    conn0 = connect("postgres")
    conn0.autocommit = True
    cur0 = conn0.cursor()
    cur0.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if not cur0.fetchone():
        info(f"Creating database '{db_name}'...")
        cur0.execute(f'CREATE DATABASE "{db_name}"')
        ok(f"Database '{db_name}' created")
    else:
        ok(f"Database '{db_name}' already exists")
    cur0.close()
    conn0.close()

    conn = connect(db_name)
    cur = conn.cursor()

    info("Creating PostgreSQL tables and views...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sku_forecasts (
            id              BIGSERIAL       PRIMARY KEY,
            forecast_date   DATE            NOT NULL,
            sku_id          VARCHAR(20)     NOT NULL,
            horizon_days    SMALLINT        NOT NULL DEFAULT 30,
            p10_total       NUMERIC(12,2),
            p50_total       NUMERIC(12,2),
            p90_total       NUMERIC(12,2),
            p50_daily       NUMERIC(10,2),
            forecast_source VARCHAR(50)     NOT NULL DEFAULT 'statistical',
            model_version   VARCHAR(50),
            created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            UNIQUE (forecast_date, sku_id, horizon_days, forecast_source)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecasts_sku  ON sku_forecasts (sku_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecasts_date ON sku_forecasts (forecast_date)"
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS inventory_alerts (
            id                BIGSERIAL       PRIMARY KEY,
            alert_date        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            sku_id            VARCHAR(20)     NOT NULL,
            alert_type        VARCHAR(30)     NOT NULL,
            days_of_supply    NUMERIC(8,2),
            current_inventory INT,
            avg_daily_demand  NUMERIC(10,2),
            lead_time_days    SMALLINT,
            recommended_action TEXT,
            resolved          BOOLEAN         NOT NULL DEFAULT FALSE,
            resolved_at       TIMESTAMPTZ
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_alerts_sku      ON inventory_alerts (sku_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON inventory_alerts (resolved)"
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS demand_anomalies (
            id              BIGSERIAL       PRIMARY KEY,
            detected_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            sku_id          VARCHAR(20)     NOT NULL,
            anomaly_date    DATE            NOT NULL,
            observed_demand INT             NOT NULL,
            expected_demand NUMERIC(10,2),
            z_score         NUMERIC(6,3),
            anomaly_type    VARCHAR(30),
            notes           TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS monthly_kpis (
            id                  BIGSERIAL       PRIMARY KEY,
            year_month          CHAR(7)         NOT NULL,
            sku_id              VARCHAR(20)     NOT NULL,
            category            VARCHAR(100),
            brand               VARCHAR(100),
            total_demand        BIGINT,
            avg_daily_demand    NUMERIC(10,2),
            avg_inventory       NUMERIC(12,2),
            min_inventory       INT,
            stockout_days       SMALLINT        DEFAULT 0,
            revenue_est_usd     NUMERIC(14,2),
            fill_rate_pct       NUMERIC(5,2),
            UNIQUE (year_month, sku_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS supplier_risk_scores (
            id                  BIGSERIAL       PRIMARY KEY,
            score_date          DATE            NOT NULL DEFAULT CURRENT_DATE,
            supplier_name       VARCHAR(150)    NOT NULL,
            risk_score          NUMERIC(5,2),
            on_time_score       NUMERIC(5,2),
            quality_score       NUMERIC(5,2),
            lead_time_variance  NUMERIC(5,2),
            concentration_risk  NUMERIC(5,2),
            notes               TEXT,
            UNIQUE (score_date, supplier_name)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_query_log (
            id              BIGSERIAL       PRIMARY KEY,
            session_id      UUID            NOT NULL DEFAULT gen_random_uuid(),
            queried_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
            provider        VARCHAR(50),
            model           VARCHAR(100),
            user_query      TEXT,
            tools_called    TEXT[],
            response_tokens INT,
            duration_ms     INT
        )
    """)

    cur.execute("""
        CREATE OR REPLACE VIEW forecast_accuracy AS
        SELECT sku_id, forecast_date, horizon_days,
               p50_daily AS predicted_daily, forecast_source, created_at
        FROM sku_forecasts ORDER BY created_at DESC
    """)

    cur.execute("""
        CREATE OR REPLACE VIEW active_alerts AS
        SELECT * FROM inventory_alerts
        WHERE resolved = FALSE ORDER BY alert_date DESC
    """)

    conn.commit()
    ok("All PostgreSQL tables and views created")

    info("Seeding forecasts for all 60 SKUs (last 3 months)...")
    import numpy as np

    sku_ids = df["sku_id"].unique().tolist()
    forecast_rows = []
    today = date.today()
    for sku_id in sku_ids:
        sku_df = df[df["sku_id"] == sku_id].sort_values("date")
        recent = sku_df["demand"].values[-60:]
        avg = float(np.mean(recent))
        std = float(np.std(recent))
        for days_ago in [0, 30, 60, 90]:
            fd = today - timedelta(days=days_ago)
            p50 = round(avg * 30, 2)  # BUG-040 fix: no artificial 1.02 upward bias
            p10 = round(max(0, p50 - 1.65 * std * 30**0.5), 2)
            p90 = round(p50 + 1.65 * std * 30**0.5, 2)
            forecast_rows.append(
                (
                    fd,
                    sku_id,
                    30,
                    p10,
                    p50,
                    p90,
                    round(avg, 2),
                    "statistical",
                    "v1.0",
                )
            )
    execute_values(
        cur,
        """
        INSERT INTO sku_forecasts
            (forecast_date, sku_id, horizon_days, p10_total, p50_total,
             p90_total, p50_daily, forecast_source, model_version)
        VALUES %s
        ON CONFLICT (forecast_date, sku_id, horizon_days, forecast_source) DO NOTHING
    """,
        forecast_rows,
    )
    conn.commit()
    ok(f"Seeded {len(forecast_rows)} forecast rows")

    info("Seeding sample inventory alerts...")
    import random

    random.seed(99)
    alert_skus = random.sample(sku_ids, 12)
    alert_rows = []
    for sku_id in alert_skus:
        sku_df = df[df["sku_id"] == sku_id].sort_values("date")
        inv = int(sku_df["inventory"].iloc[-1])
        lt = int(sku_df["lead_time_days"].iloc[-1])
        avg_d = float(sku_df["demand"].tail(30).mean())
        dos = round(inv / avg_d if avg_d > 0 else 0, 1)
        alert_type = "CRITICAL" if dos < lt else "WARNING"
        action = (
            f"Order {int(avg_d * 45):,} units immediately from supplier."
            if alert_type == "CRITICAL"
            else f"Schedule reorder within 7 days. Order {int(avg_d * 30):,} units."
        )
        alert_rows.append(
            (sku_id, alert_type, dos, inv, round(avg_d, 2), lt, action, False)
        )
    execute_values(
        cur,
        """
        INSERT INTO inventory_alerts
            (sku_id, alert_type, days_of_supply, current_inventory,
             avg_daily_demand, lead_time_days, recommended_action, resolved)
        VALUES %s
    """,
        alert_rows,
    )
    conn.commit()
    ok(f"Seeded {len(alert_rows)} inventory alerts")

    info("Computing and seeding monthly KPIs (Jan 2023 to Dec 2024)...")
    df2 = df.copy()
    df2["year_month"] = df2["date"].dt.to_period("M").astype(str)
    # Count actual days per (year_month, sku_id) group for correct fill_rate denominator
    df2["_one"] = 1
    kpi_df = (
        df2.groupby(["year_month", "sku_id"])
        .agg(
            total_demand=("demand", "sum"),
            avg_daily_demand=("demand", "mean"),
            avg_inventory=("inventory", "mean"),
            min_inventory=("inventory", "min"),
            stockout_days=("inventory", lambda x: int((x == 0).sum())),
            day_count=("_one", "sum"),
            price=("price_usd", "first"),
            category=("category", "first")
            if "category" in df2.columns
            else ("sku_id", lambda x: ""),
            brand=("brand", "first")
            if "brand" in df2.columns
            else ("sku_id", lambda x: ""),
        )
        .reset_index()
    )
    kpi_df["revenue_est_usd"] = (kpi_df["total_demand"] * kpi_df["price"]).round(2)
    # fill_rate uses actual day count per month — not hardcoded 30 (H-03 fix)
    kpi_df["fill_rate_pct"] = (
        ((kpi_df["day_count"] - kpi_df["stockout_days"]) / kpi_df["day_count"] * 100)
        .clip(lower=0, upper=100)
        .round(2)
    )

    kpi_rows = [
        (
            r.year_month,
            r.sku_id,
            str(r.category) if hasattr(r, "category") else "",
            str(r.brand) if hasattr(r, "brand") else "",
            int(r.total_demand),
            round(r.avg_daily_demand, 2),
            round(r.avg_inventory, 2),
            int(r.min_inventory),
            int(r.stockout_days),
            float(r.revenue_est_usd),
            float(r.fill_rate_pct),
        )
        for r in kpi_df.itertuples()
    ]
    execute_values(
        cur,
        """
        INSERT INTO monthly_kpis
            (year_month, sku_id, category, brand, total_demand, avg_daily_demand,
             avg_inventory, min_inventory, stockout_days, revenue_est_usd, fill_rate_pct)
        VALUES %s
        ON CONFLICT (year_month, sku_id) DO NOTHING
    """,
        kpi_rows,
    )
    conn.commit()
    ok(
        f"Seeded {len(kpi_rows)} monthly KPI rows ({kpi_df['year_month'].nunique()} months × {kpi_df['sku_id'].nunique()} SKUs)"
    )

    info("Seeding supplier risk scores...")
    risk_rows = [
        (
            date.today(),
            "PawsSupply Co",
            12.5,
            97.0,
            94.0,
            1.5,
            28.0,
            "Low risk. Primary distributor.",
        ),
        (
            date.today(),
            "NutriPet Inc",
            25.0,
            90.0,
            90.0,
            4.0,
            18.0,
            "Medium risk. High MOQ limits flexibility.",
        ),
        (
            date.today(),
            "GlobalPet Dist",
            38.0,
            82.0,
            82.0,
            8.0,
            15.0,
            "Medium-high risk. Customs delays.",
        ),
        (
            date.today(),
            "TreatWorld LLC",
            62.0,
            74.0,
            72.0,
            12.0,
            22.0,
            "High risk. Past quality incident.",
        ),
        (
            date.today(),
            "HealthPet Labs",
            15.0,
            95.0,
            92.0,
            2.0,
            12.0,
            "Low risk. FDA certified.",
        ),
        (
            date.today(),
            "VetPharma Supply",
            28.0,
            88.0,
            96.0,
            5.0,
            8.0,
            "Medium risk. Long international lead time.",
        ),
        (
            date.today(),
            "PetEssentials",
            10.0,
            98.0,
            88.0,
            1.0,
            20.0,
            "Low risk. Best OTD in portfolio.",
        ),
        (
            date.today(),
            "ToyPet Factory",
            72.0,
            68.0,
            68.0,
            18.0,
            25.0,
            "High risk. Low OTD, high MOQ.",
        ),
        (
            date.today(),
            "RawPet Foods",
            18.0,
            95.0,
            98.0,
            2.0,
            5.0,
            "Low risk. Premium quality, cold chain capable.",
        ),
    ]
    execute_values(
        cur,
        """
        INSERT INTO supplier_risk_scores
            (score_date, supplier_name, risk_score, on_time_score, quality_score,
             lead_time_variance, concentration_risk, notes)
        VALUES %s
        ON CONFLICT (score_date, supplier_name) DO NOTHING
    """,
        risk_rows,
    )
    conn.commit()
    ok(f"Seeded {len(risk_rows)} supplier risk scores")

    cur.close()
    conn.close()
    ok("PostgreSQL setup complete")


def print_summary() -> None:
    step("Step 4 of 4 -- Summary")
    cfg_m = get_mysql_cfg()
    cfg_p = get_pg_cfg()
    print(f"""
  Everything is set up. Here is what was created:

  MySQL ({cfg_m["host"]}:{cfg_m["port"]} / {cfg_m["db"]}):
    skus                  60 rows  (one per SKU)
    suppliers              9 rows  (all 9 suppliers)
    daily_demand      87,660 rows  (Jan 2021 - Dec 2024)
    reorder_events       120 rows  (sample purchase orders)
    supplier_performance 108 rows  (quarterly reviews 2022-2024)

  PostgreSQL ({cfg_p["host"]}:{cfg_p["port"]} / {cfg_p["db"]}):
    sku_forecasts        240 rows  (4 forecast dates x 60 SKUs)
    inventory_alerts      12 rows  (sample critical/warning alerts)
    monthly_kpis       2,880 rows  (48 months x 60 SKUs)
    supplier_risk_scores  9 rows  (one per supplier)
    demand_anomalies       0 rows  (populated by the agent at runtime)
    agent_query_log        0 rows  (populated by the agent at runtime)

  Now run the app:
    python gradio_app.py

  Then open: http://localhost:7860
    - Click "Database Settings" and hit "Test MySQL Connection"
      and "Test PostgreSQL Connection" to confirm both are live.
    - Pick your LLM provider and paste your API key in the UI.
    - Start chatting!
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pet Store SCM database setup — works with local or cloud databases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local (uses .env credentials):
    python db/setup.py

  Cloud MySQL only (Railway example):
    python db/setup.py --mysql-only \\
      --mysql-host containers-us-west-123.railway.app \\
      --mysql-port 6543 --mysql-user root \\
      --mysql-password mypassword --mysql-db pet_store_scm

  Cloud PostgreSQL only (Supabase example):
    python db/setup.py --postgres-only \\
      --pg-host db.abcdefg.supabase.co \\
      --pg-port 5432 --pg-user postgres \\
      --pg-password mypassword --pg-db postgres

  Both cloud (Railway MySQL + Supabase PostgreSQL):
    python db/setup.py \\
      --mysql-host ... --mysql-port ... --mysql-user ... --mysql-password ... \\
      --pg-host    ... --pg-port    ... --pg-user    ... --pg-password    ...
        """,
    )
    parser.add_argument("--mysql-only", action="store_true", help="Only set up MySQL")
    parser.add_argument(
        "--postgres-only", action="store_true", help="Only set up PostgreSQL"
    )
    parser.add_argument(
        "--skip-data", action="store_true", help="Skip CSV generation, use existing CSV"
    )

    # Optional credential overrides — if provided, these override .env values
    parser.add_argument(
        "--mysql-host", default=None, help="MySQL host (overrides .env)"
    )
    parser.add_argument(
        "--mysql-port", default=None, help="MySQL port (overrides .env)"
    )
    parser.add_argument(
        "--mysql-user", default=None, help="MySQL username (overrides .env)"
    )
    parser.add_argument(
        "--mysql-password", default=None, help="MySQL password (overrides .env)"
    )
    parser.add_argument(
        "--mysql-db", default=None, help="MySQL database name (overrides .env)"
    )
    parser.add_argument(
        "--pg-host", default=None, help="PostgreSQL host (overrides .env)"
    )
    parser.add_argument(
        "--pg-port", default=None, help="PostgreSQL port (overrides .env)"
    )
    parser.add_argument(
        "--pg-user", default=None, help="PostgreSQL username (overrides .env)"
    )
    parser.add_argument(
        "--pg-password", default=None, help="PostgreSQL password (overrides .env)"
    )
    parser.add_argument(
        "--pg-db", default=None, help="PostgreSQL database name (overrides .env)"
    )

    args = parser.parse_args()

    # Override env vars if CLI args were provided — this way get_mysql_cfg()
    # and get_pg_cfg() will automatically pick them up
    if args.mysql_host:
        os.environ["MYSQL_HOST"] = args.mysql_host
    if args.mysql_port:
        os.environ["MYSQL_PORT"] = args.mysql_port
    if args.mysql_user:
        os.environ["MYSQL_USER"] = args.mysql_user
    if args.mysql_password:
        os.environ["MYSQL_PASSWORD"] = args.mysql_password
    if args.mysql_db:
        os.environ["MYSQL_DB"] = args.mysql_db
    if args.pg_host:
        os.environ["PG_HOST"] = args.pg_host
    if args.pg_port:
        os.environ["PG_PORT"] = args.pg_port
    if args.pg_user:
        os.environ["PG_USER"] = args.pg_user
    if args.pg_password:
        os.environ["PG_PASSWORD"] = args.pg_password
    if args.pg_db:
        os.environ["PG_DB"] = args.pg_db

    print("\nPet Store SCM -- Database Setup\n")

    cfg_m = get_mysql_cfg()
    cfg_p = get_pg_cfg()
    print(
        f"  MySQL target:      {cfg_m['user']}@{cfg_m['host']}:{cfg_m['port']}/{cfg_m['db']}"
    )
    print(
        f"  PostgreSQL target: {cfg_p['user']}@{cfg_p['host']}:{cfg_p['port']}/{cfg_p['db']}"
    )

    import pandas as pd

    if args.skip_data:
        csv_path = BASE_DIR / "data" / "pet_store_supply_chain.csv"
        if not csv_path.exists():
            print(
                "\nERROR: --skip-data specified but CSV not found. Run without --skip-data first."
            )
            sys.exit(1)
        df = pd.read_csv(csv_path, parse_dates=["date"])
        ok(f"Using existing CSV ({len(df):,} rows)")
    else:
        df = generate_data()

    if args.postgres_only:
        setup_postgres(df)
    elif args.mysql_only:
        setup_mysql(df)
    else:
        setup_mysql(df)
        setup_postgres(df)

    print_summary()


if __name__ == "__main__":
    main()
