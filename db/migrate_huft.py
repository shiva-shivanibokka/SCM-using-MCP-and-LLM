"""
db/migrate_huft.py
──────────────────
Migrates ALL databases (local + cloud MySQL and PostgreSQL) to the new
HUFT schema and loads all 9 synthetic CSV datasets.

Run once:
    python db/migrate_huft.py

What it does:
  1. Drops old generic pet-store tables
  2. Creates new HUFT tables (products, stores, customers, promotions,
     daily_demand, sales_transactions, returns, supplier_performance,
     cold_chain + PostgreSQL analytics tables)
  3. Loads all 9 CSV files into every database
  4. Targets: local MySQL, cloud MySQL (Railway), local PostgreSQL,
     cloud PostgreSQL (Railway)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"

# ── CSV paths ──────────────────────────────────────────────────────────────
CSV = {
    "products": DATA_DIR / "huft_products.csv",
    "stores": DATA_DIR / "huft_stores.csv",
    "customers": DATA_DIR / "huft_customers.csv",
    "promotions": DATA_DIR / "huft_promotions.csv",
    "daily_demand": DATA_DIR / "huft_daily_demand.csv",
    "sales_transactions": DATA_DIR / "huft_sales_transactions.csv",
    "returns": DATA_DIR / "huft_returns.csv",
    "supplier_performance": DATA_DIR / "huft_supplier_performance.csv",
    "cold_chain": DATA_DIR / "huft_cold_chain.csv",
    "store_daily_inventory": DATA_DIR / "store_daily_inventory.csv",
}

# ── Connection configs ─────────────────────────────────────────────────────
MYSQL_LOCAL = dict(
    host=os.getenv("MYSQL_HOST", "localhost"),
    port=int(os.getenv("MYSQL_PORT", 3306)),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DB", "pet_store_scm"),
)
MYSQL_CLOUD = dict(
    host=os.getenv("MYSQL_CLOUD_HOST", ""),
    port=int(os.getenv("MYSQL_CLOUD_PORT", 3306)),
    user=os.getenv("MYSQL_CLOUD_USER", "root"),
    password=os.getenv("MYSQL_CLOUD_PASSWORD", ""),
    database=os.getenv("MYSQL_CLOUD_DB", "railway"),
)
PG_LOCAL = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=int(os.getenv("PG_PORT", 5432)),
    user=os.getenv("PG_USER", "postgres"),
    password=os.getenv("PG_PASSWORD", ""),
    dbname=os.getenv("PG_DB", "pet_store_scm"),
)
PG_CLOUD = dict(
    host=os.getenv("PG_CLOUD_HOST", ""),
    port=int(os.getenv("PG_CLOUD_PORT", 5432)),
    user=os.getenv("PG_CLOUD_USER", "postgres"),
    password=os.getenv("PG_CLOUD_PASSWORD", ""),
    dbname=os.getenv("PG_CLOUD_DB", "railway"),
)

# ══════════════════════════════════════════════════════════════════════════════
#  MySQL DDL
# ══════════════════════════════════════════════════════════════════════════════

MYSQL_DROP = """
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS store_daily_inventory;
DROP TABLE IF EXISTS cold_chain;
DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS sales_transactions;
DROP TABLE IF EXISTS promotions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS daily_demand;
DROP TABLE IF EXISTS supplier_performance;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS stores;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS skus;
DROP TABLE IF EXISTS reorder_events;
SET FOREIGN_KEY_CHECKS = 1;
"""

MYSQL_DDL = """  # kept for reference only
"""

_MYSQL_TABLES = [
    """CREATE TABLE IF NOT EXISTS products (
    sku_id              VARCHAR(30)     PRIMARY KEY,
    name                VARCHAR(255)    NOT NULL,
    brand               VARCHAR(100)    NOT NULL,
    brand_type          VARCHAR(50)     NOT NULL,
    category            VARCHAR(100)    NOT NULL,
    subcategory         VARCHAR(100)    NOT NULL,
    pet_type            VARCHAR(50)     NOT NULL,
    life_stage          VARCHAR(50)     NOT NULL,
    breed_suitability   VARCHAR(100),
    weight_kg           DECIMAL(8,3),
    price_inr           DECIMAL(10,2)   NOT NULL,
    cost_inr            DECIMAL(10,2)   NOT NULL,
    supplier            VARCHAR(150)    NOT NULL,
    lead_time_days      TINYINT UNSIGNED NOT NULL,
    base_demand         SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    is_cold_chain       BOOLEAN         NOT NULL DEFAULT FALSE,
    margin_pct          DECIMAL(5,2),
    min_age_months      TINYINT UNSIGNED,
    max_age_months      TINYINT UNSIGNED,
    INDEX idx_category  (category),
    INDEX idx_brand     (brand),
    INDEX idx_pet_type  (pet_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS suppliers (
    supplier_name           VARCHAR(150)    PRIMARY KEY,
    contact_email           VARCHAR(255),
    country                 VARCHAR(100),
    on_time_delivery_pct    DECIMAL(5,2),
    quality_rating          DECIMAL(3,1),
    min_order_qty           SMALLINT UNSIGNED,
    emergency_capable       BOOLEAN DEFAULT FALSE,
    notes                   TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS stores (
    store_id        VARCHAR(10)     PRIMARY KEY,
    city            VARCHAR(100)    NOT NULL,
    state           VARCHAR(100),
    region          VARCHAR(50)     NOT NULL,
    store_type      VARCHAR(30)     NOT NULL,
    opened_year     SMALLINT UNSIGNED,
    size_sqft       INT UNSIGNED,
    has_spa         BOOLEAN         NOT NULL DEFAULT FALSE,
    INDEX idx_region    (region),
    INDEX idx_city      (city)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS customers (
    customer_id             VARCHAR(20)     PRIMARY KEY,
    city                    VARCHAR(100)    NOT NULL,
    segment                 VARCHAR(50)     NOT NULL,
    joined_date             DATE            NOT NULL,
    pet_type                VARCHAR(20)     NOT NULL,
    total_orders            SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    lifetime_value_inr      DECIMAL(12,2),
    channel_preference      VARCHAR(20),
    is_spa_customer         BOOLEAN         NOT NULL DEFAULT FALSE,
    breed                   VARCHAR(100),
    INDEX idx_segment   (segment),
    INDEX idx_city      (city)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS promotions (
    promo_id            VARCHAR(20)     PRIMARY KEY,
    name                VARCHAR(255)    NOT NULL,
    start_date          DATE            NOT NULL,
    end_date            DATE            NOT NULL,
    discount_pct        DECIMAL(5,2)    NOT NULL DEFAULT 0,
    channel             VARCHAR(30)     NOT NULL DEFAULT 'All',
    target_category     VARCHAR(100)    NOT NULL DEFAULT 'All',
    budget_inr          DECIMAL(12,2),
    duration_days       SMALLINT UNSIGNED,
    revenue_generated_inr DECIMAL(14,2),
    units_sold          INT UNSIGNED,
    INDEX idx_dates (start_date, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS daily_demand (
    id              BIGINT UNSIGNED     AUTO_INCREMENT PRIMARY KEY,
    record_date     DATE                NOT NULL,
    sku_id          VARCHAR(30)         NOT NULL,
    name            VARCHAR(255),
    brand           VARCHAR(100),
    brand_type      VARCHAR(50),
    category        VARCHAR(100)        NOT NULL,
    subcategory     VARCHAR(100),
    pet_type        VARCHAR(50),
    life_stage      VARCHAR(50),
    supplier        VARCHAR(150),
    demand          MEDIUMINT UNSIGNED  NOT NULL DEFAULT 0,
    inventory       MEDIUMINT UNSIGNED  NOT NULL DEFAULT 0,
    lead_time_days  TINYINT UNSIGNED    NOT NULL DEFAULT 7,
    price_inr       DECIMAL(10,2),
    cost_inr        DECIMAL(10,2),
    margin_pct      DECIMAL(5,2),
    is_cold_chain   BOOLEAN             NOT NULL DEFAULT FALSE,
    UNIQUE KEY uq_date_sku (record_date, sku_id),
    INDEX idx_sku       (sku_id),
    INDEX idx_date      (record_date),
    INDEX idx_category  (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS sales_transactions (
    txn_id              VARCHAR(20)     PRIMARY KEY,
    txn_date            DATE            NOT NULL,
    sku_id              VARCHAR(30)     NOT NULL,
    brand               VARCHAR(100),
    category            VARCHAR(100),
    quantity            TINYINT UNSIGNED NOT NULL DEFAULT 1,
    unit_price_inr      DECIMAL(10,2)   NOT NULL,
    discount_pct        DECIMAL(5,2)    NOT NULL DEFAULT 0,
    net_revenue_inr     DECIMAL(12,2)   NOT NULL,
    gross_margin_inr    DECIMAL(12,2),
    channel             VARCHAR(20)     NOT NULL,
    city                VARCHAR(100),
    customer_segment    VARCHAR(50),
    store_id            VARCHAR(10),
    INDEX idx_date      (txn_date),
    INDEX idx_sku       (sku_id),
    INDEX idx_channel   (channel),
    INDEX idx_city      (city)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS returns (
    return_id           VARCHAR(20)     PRIMARY KEY,
    original_txn_id     VARCHAR(20),
    sku_id              VARCHAR(30)     NOT NULL,
    category            VARCHAR(100),
    brand               VARCHAR(100),
    return_date         DATE            NOT NULL,
    quantity_returned   TINYINT UNSIGNED NOT NULL DEFAULT 1,
    return_reason       VARCHAR(100),
    refund_inr          DECIMAL(10,2),
    channel             VARCHAR(20),
    city                VARCHAR(100),
    INDEX idx_sku       (sku_id),
    INDEX idx_date      (return_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS supplier_performance (
    id                      BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    supplier_name           VARCHAR(150)    NOT NULL,
    review_month            CHAR(7)         NOT NULL,
    on_time_delivery_pct    DECIMAL(5,2),
    defect_rate_pct         DECIMAL(5,2),
    fill_rate_pct           DECIMAL(5,2),
    lead_time_actual_days   TINYINT UNSIGNED,
    quality_rating          DECIMAL(3,1),
    notes                   TEXT,
    UNIQUE KEY uq_supplier_month (supplier_name, review_month),
    INDEX idx_supplier  (supplier_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS cold_chain (
    id                          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    record_date                 DATE            NOT NULL,
    sku_id                      VARCHAR(30)     NOT NULL,
    name                        VARCHAR(255),
    temp_celsius                DECIMAL(5,2),
    target_min_c                DECIMAL(4,2),
    target_max_c                DECIMAL(4,2),
    temp_breach                 BOOLEAN         NOT NULL DEFAULT FALSE,
    units_in_cold_storage       SMALLINT UNSIGNED,
    expiry_date                 DATE,
    shelf_life_days_remaining   TINYINT UNSIGNED,
    units_at_risk_of_expiry     SMALLINT UNSIGNED DEFAULT 0,
    batch_id                    VARCHAR(50),
    INDEX idx_sku       (sku_id),
    INDEX idx_date      (record_date),
    INDEX idx_breach    (temp_breach)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS store_daily_inventory (
    id              BIGINT UNSIGNED     AUTO_INCREMENT PRIMARY KEY,
    record_date     DATE                NOT NULL,
    store_id        VARCHAR(10)         NOT NULL,
    city            VARCHAR(100)        NOT NULL,
    state           VARCHAR(100),
    region          VARCHAR(50)         NOT NULL,
    store_type      VARCHAR(30)         NOT NULL,
    sku_id          VARCHAR(30)         NOT NULL,
    name            VARCHAR(255),
    category        VARCHAR(100)        NOT NULL,
    brand           VARCHAR(100),
    demand          SMALLINT UNSIGNED   NOT NULL DEFAULT 0,
    inventory       INT UNSIGNED        NOT NULL DEFAULT 0,
    lead_time_days  TINYINT UNSIGNED    NOT NULL DEFAULT 7,
    days_of_supply  DECIMAL(8,1)        NOT NULL DEFAULT 0,
    risk_status     VARCHAR(10)         NOT NULL DEFAULT 'OK',
    price_inr       DECIMAL(10,2),
    cost_inr        DECIMAL(10,2),
    UNIQUE KEY uq_store_date_sku (record_date, store_id, sku_id),
    INDEX idx_store     (store_id),
    INDEX idx_date      (record_date),
    INDEX idx_sku       (sku_id),
    INDEX idx_city      (city),
    INDEX idx_region    (region),
    INDEX idx_risk      (risk_status),
    INDEX idx_dos       (days_of_supply),
    INDEX idx_category  (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]  # end _MYSQL_TABLES


_ORIG_MYSQL_DDL = """
-- ── products ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    sku_id              VARCHAR(30)     PRIMARY KEY,
    name                VARCHAR(255)    NOT NULL,
    brand               VARCHAR(100)    NOT NULL,
    brand_type          VARCHAR(50)     NOT NULL,
    category            VARCHAR(100)    NOT NULL,
    subcategory         VARCHAR(100)    NOT NULL,
    pet_type            VARCHAR(50)     NOT NULL,
    life_stage          VARCHAR(50)     NOT NULL,
    breed_suitability   VARCHAR(100),
    weight_kg           DECIMAL(8,3),
    price_inr           DECIMAL(10,2)   NOT NULL,
    cost_inr            DECIMAL(10,2)   NOT NULL,
    supplier            VARCHAR(150)    NOT NULL,
    lead_time_days      TINYINT UNSIGNED NOT NULL,
    base_demand         SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    is_cold_chain       BOOLEAN         NOT NULL DEFAULT FALSE,
    margin_pct          DECIMAL(5,2),
    min_age_months      TINYINT UNSIGNED,
    max_age_months      TINYINT UNSIGNED,
    INDEX idx_category  (category),
    INDEX idx_brand     (brand),
    INDEX idx_pet_type  (pet_type),
    INDEX idx_life_stage(life_stage)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── suppliers ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_name           VARCHAR(150)    PRIMARY KEY,
    contact_email           VARCHAR(255),
    country                 VARCHAR(100),
    on_time_delivery_pct    DECIMAL(5,2),
    quality_rating          DECIMAL(3,1),
    min_order_qty           SMALLINT UNSIGNED,
    emergency_capable       BOOLEAN DEFAULT FALSE,
    notes                   TEXT,
    INDEX idx_country (country)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── stores ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stores (
    store_id        VARCHAR(10)     PRIMARY KEY,
    city            VARCHAR(100)    NOT NULL,
    state           VARCHAR(100),
    region          VARCHAR(50)     NOT NULL,
    store_type      VARCHAR(30)     NOT NULL,
    opened_year     SMALLINT UNSIGNED,
    size_sqft       INT UNSIGNED,
    has_spa         BOOLEAN         NOT NULL DEFAULT FALSE,
    INDEX idx_region    (region),
    INDEX idx_city      (city),
    INDEX idx_store_type(store_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── customers ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id             VARCHAR(20)     PRIMARY KEY,
    city                    VARCHAR(100)    NOT NULL,
    segment                 VARCHAR(50)     NOT NULL,
    joined_date             DATE            NOT NULL,
    pet_type                VARCHAR(20)     NOT NULL,
    total_orders            SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    lifetime_value_inr      DECIMAL(12,2),
    channel_preference      VARCHAR(20),
    is_spa_customer         BOOLEAN         NOT NULL DEFAULT FALSE,
    breed                   VARCHAR(100),
    INDEX idx_segment   (segment),
    INDEX idx_city      (city),
    INDEX idx_pet_type  (pet_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── promotions ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS promotions (
    promo_id            VARCHAR(20)     PRIMARY KEY,
    name                VARCHAR(255)    NOT NULL,
    start_date          DATE            NOT NULL,
    end_date            DATE            NOT NULL,
    discount_pct        DECIMAL(5,2)    NOT NULL DEFAULT 0,
    channel             VARCHAR(30)     NOT NULL DEFAULT 'All',
    target_category     VARCHAR(100)    NOT NULL DEFAULT 'All',
    budget_inr          DECIMAL(12,2),
    duration_days       SMALLINT UNSIGNED,
    revenue_generated_inr DECIMAL(14,2),
    units_sold          INT UNSIGNED,
    INDEX idx_dates (start_date, end_date),
    INDEX idx_channel (channel)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── daily_demand ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_demand (
    id              BIGINT UNSIGNED     AUTO_INCREMENT PRIMARY KEY,
    record_date     DATE                NOT NULL,
    sku_id          VARCHAR(30)         NOT NULL,
    name            VARCHAR(255),
    brand           VARCHAR(100),
    brand_type      VARCHAR(50),
    category        VARCHAR(100)        NOT NULL,
    subcategory     VARCHAR(100),
    pet_type        VARCHAR(50),
    life_stage      VARCHAR(50),
    supplier        VARCHAR(150),
    demand          MEDIUMINT UNSIGNED  NOT NULL DEFAULT 0,
    inventory       MEDIUMINT UNSIGNED  NOT NULL DEFAULT 0,
    lead_time_days  TINYINT UNSIGNED    NOT NULL DEFAULT 7,
    price_inr       DECIMAL(10,2),
    cost_inr        DECIMAL(10,2),
    margin_pct      DECIMAL(5,2),
    is_cold_chain   BOOLEAN             NOT NULL DEFAULT FALSE,
    UNIQUE KEY uq_date_sku (record_date, sku_id),
    INDEX idx_sku       (sku_id),
    INDEX idx_date      (record_date),
    INDEX idx_category  (category),
    INDEX idx_brand     (brand)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── sales_transactions ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sales_transactions (
    txn_id              VARCHAR(20)     PRIMARY KEY,
    txn_date            DATE            NOT NULL,
    sku_id              VARCHAR(30)     NOT NULL,
    brand               VARCHAR(100),
    category            VARCHAR(100),
    quantity            TINYINT UNSIGNED NOT NULL DEFAULT 1,
    unit_price_inr      DECIMAL(10,2)   NOT NULL,
    discount_pct        DECIMAL(5,2)    NOT NULL DEFAULT 0,
    net_revenue_inr     DECIMAL(12,2)   NOT NULL,
    gross_margin_inr    DECIMAL(12,2),
    channel             VARCHAR(20)     NOT NULL,
    city                VARCHAR(100),
    customer_segment    VARCHAR(50),
    store_id            VARCHAR(10),
    INDEX idx_date      (txn_date),
    INDEX idx_sku       (sku_id),
    INDEX idx_channel   (channel),
    INDEX idx_city      (city),
    INDEX idx_segment   (customer_segment),
    INDEX idx_store     (store_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── returns ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS returns (
    return_id           VARCHAR(20)     PRIMARY KEY,
    original_txn_id     VARCHAR(20),
    sku_id              VARCHAR(30)     NOT NULL,
    category            VARCHAR(100),
    brand               VARCHAR(100),
    return_date         DATE            NOT NULL,
    quantity_returned   TINYINT UNSIGNED NOT NULL DEFAULT 1,
    return_reason       VARCHAR(100),
    refund_inr          DECIMAL(10,2),
    channel             VARCHAR(20),
    city                VARCHAR(100),
    INDEX idx_sku       (sku_id),
    INDEX idx_date      (return_date),
    INDEX idx_reason    (return_reason)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── supplier_performance ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS supplier_performance (
    id                      BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    supplier_name           VARCHAR(150)    NOT NULL,
    review_month            CHAR(7)         NOT NULL,
    on_time_delivery_pct    DECIMAL(5,2),
    defect_rate_pct         DECIMAL(5,2),
    fill_rate_pct           DECIMAL(5,2),
    lead_time_actual_days   TINYINT UNSIGNED,
    quality_rating          DECIMAL(3,1),
    notes                   TEXT,
    UNIQUE KEY uq_supplier_month (supplier_name, review_month),
    INDEX idx_supplier  (supplier_name),
    INDEX idx_month     (review_month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── cold_chain ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cold_chain (
    id                      BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    record_date             DATE            NOT NULL,
    sku_id                  VARCHAR(30)     NOT NULL,
    name                    VARCHAR(255),
    temp_celsius            DECIMAL(5,2),
    target_min_c            DECIMAL(4,2),
    target_max_c            DECIMAL(4,2),
    temp_breach             BOOLEAN         NOT NULL DEFAULT FALSE,
    units_in_cold_storage   SMALLINT UNSIGNED,
    expiry_date             DATE,
    shelf_life_days_remaining TINYINT UNSIGNED,
    units_at_risk_of_expiry SMALLINT UNSIGNED DEFAULT 0,
    batch_id                VARCHAR(50),
    INDEX idx_sku       (sku_id),
    INDEX idx_date      (record_date),
    INDEX idx_breach    (temp_breach)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ══════════════════════════════════════════════════════════════════════════════
#  PostgreSQL DDL
# ══════════════════════════════════════════════════════════════════════════════

PG_DROP = """
DROP TABLE IF EXISTS store_daily_inventory  CASCADE;
DROP TABLE IF EXISTS cold_chain            CASCADE;
DROP TABLE IF EXISTS returns               CASCADE;
DROP TABLE IF EXISTS sales_transactions    CASCADE;
DROP TABLE IF EXISTS promotions            CASCADE;
DROP TABLE IF EXISTS customers             CASCADE;
DROP TABLE IF EXISTS daily_demand          CASCADE;
DROP TABLE IF EXISTS supplier_performance  CASCADE;
DROP TABLE IF EXISTS suppliers             CASCADE;
DROP TABLE IF EXISTS stores                CASCADE;
DROP TABLE IF EXISTS products              CASCADE;
DROP TABLE IF EXISTS sku_forecasts         CASCADE;
DROP TABLE IF EXISTS inventory_alerts      CASCADE;
DROP TABLE IF EXISTS demand_anomalies      CASCADE;
DROP TABLE IF EXISTS monthly_kpis          CASCADE;
DROP TABLE IF EXISTS supplier_risk_scores  CASCADE;
DROP TABLE IF EXISTS agent_query_log       CASCADE;
DROP VIEW  IF EXISTS forecast_accuracy     CASCADE;
DROP VIEW  IF EXISTS active_alerts         CASCADE;
"""

PG_DDL = """
-- ── products ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    sku_id              VARCHAR(30)     PRIMARY KEY,
    name                VARCHAR(255)    NOT NULL,
    brand               VARCHAR(100)    NOT NULL,
    brand_type          VARCHAR(50)     NOT NULL,
    category            VARCHAR(100)    NOT NULL,
    subcategory         VARCHAR(100)    NOT NULL,
    pet_type            VARCHAR(50)     NOT NULL,
    life_stage          VARCHAR(50)     NOT NULL,
    breed_suitability   VARCHAR(100),
    weight_kg           NUMERIC(8,3),
    price_inr           NUMERIC(10,2)   NOT NULL,
    cost_inr            NUMERIC(10,2)   NOT NULL,
    supplier            VARCHAR(150)    NOT NULL,
    lead_time_days      SMALLINT        NOT NULL,
    base_demand         SMALLINT        NOT NULL DEFAULT 0,
    is_cold_chain       BOOLEAN         NOT NULL DEFAULT FALSE,
    margin_pct          NUMERIC(5,2),
    min_age_months      SMALLINT,
    max_age_months      SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_products_category  ON products (category);
CREATE INDEX IF NOT EXISTS idx_products_brand     ON products (brand);
CREATE INDEX IF NOT EXISTS idx_products_pet_type  ON products (pet_type);

-- ── suppliers ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_name           VARCHAR(150)    PRIMARY KEY,
    contact_email           VARCHAR(255),
    country                 VARCHAR(100),
    on_time_delivery_pct    NUMERIC(5,2),
    quality_rating          NUMERIC(3,1),
    min_order_qty           SMALLINT,
    emergency_capable       BOOLEAN DEFAULT FALSE,
    notes                   TEXT
);

-- ── stores ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stores (
    store_id        VARCHAR(10)     PRIMARY KEY,
    city            VARCHAR(100)    NOT NULL,
    state           VARCHAR(100),
    region          VARCHAR(50)     NOT NULL,
    store_type      VARCHAR(30)     NOT NULL,
    opened_year     SMALLINT,
    size_sqft       INT,
    has_spa         BOOLEAN         NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_stores_region ON stores (region);
CREATE INDEX IF NOT EXISTS idx_stores_city   ON stores (city);

-- ── customers ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id             VARCHAR(20)     PRIMARY KEY,
    city                    VARCHAR(100)    NOT NULL,
    segment                 VARCHAR(50)     NOT NULL,
    joined_date             DATE            NOT NULL,
    pet_type                VARCHAR(20)     NOT NULL,
    total_orders            SMALLINT        NOT NULL DEFAULT 0,
    lifetime_value_inr      NUMERIC(12,2),
    channel_preference      VARCHAR(20),
    is_spa_customer         BOOLEAN         NOT NULL DEFAULT FALSE,
    breed                   VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_customers_segment  ON customers (segment);
CREATE INDEX IF NOT EXISTS idx_customers_city     ON customers (city);

-- ── promotions ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS promotions (
    promo_id            VARCHAR(20)     PRIMARY KEY,
    name                VARCHAR(255)    NOT NULL,
    start_date          DATE            NOT NULL,
    end_date            DATE            NOT NULL,
    discount_pct        NUMERIC(5,2)    NOT NULL DEFAULT 0,
    channel             VARCHAR(30)     NOT NULL DEFAULT 'All',
    target_category     VARCHAR(100)    NOT NULL DEFAULT 'All',
    budget_inr          NUMERIC(12,2),
    duration_days       SMALLINT,
    revenue_generated_inr NUMERIC(14,2),
    units_sold          INT
);
CREATE INDEX IF NOT EXISTS idx_promotions_dates ON promotions (start_date, end_date);

-- ── daily_demand ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_demand (
    id              BIGSERIAL           PRIMARY KEY,
    record_date     DATE                NOT NULL,
    sku_id          VARCHAR(30)         NOT NULL,
    name            VARCHAR(255),
    brand           VARCHAR(100),
    brand_type      VARCHAR(50),
    category        VARCHAR(100)        NOT NULL,
    subcategory     VARCHAR(100),
    pet_type        VARCHAR(50),
    life_stage      VARCHAR(50),
    supplier        VARCHAR(150),
    demand          INT                 NOT NULL DEFAULT 0,
    inventory       INT                 NOT NULL DEFAULT 0,
    lead_time_days  SMALLINT            NOT NULL DEFAULT 7,
    price_inr       NUMERIC(10,2),
    cost_inr        NUMERIC(10,2),
    margin_pct      NUMERIC(5,2),
    is_cold_chain   BOOLEAN             NOT NULL DEFAULT FALSE,
    UNIQUE (record_date, sku_id)
);
CREATE INDEX IF NOT EXISTS idx_demand_sku      ON daily_demand (sku_id);
CREATE INDEX IF NOT EXISTS idx_demand_date     ON daily_demand (record_date);
CREATE INDEX IF NOT EXISTS idx_demand_category ON daily_demand (category);

-- ── sales_transactions ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sales_transactions (
    txn_id              VARCHAR(20)     PRIMARY KEY,
    txn_date            DATE            NOT NULL,
    sku_id              VARCHAR(30)     NOT NULL,
    brand               VARCHAR(100),
    category            VARCHAR(100),
    quantity            SMALLINT        NOT NULL DEFAULT 1,
    unit_price_inr      NUMERIC(10,2)   NOT NULL,
    discount_pct        NUMERIC(5,2)    NOT NULL DEFAULT 0,
    net_revenue_inr     NUMERIC(12,2)   NOT NULL,
    gross_margin_inr    NUMERIC(12,2),
    channel             VARCHAR(20)     NOT NULL,
    city                VARCHAR(100),
    customer_segment    VARCHAR(50),
    store_id            VARCHAR(10)
);
CREATE INDEX IF NOT EXISTS idx_txn_date     ON sales_transactions (txn_date);
CREATE INDEX IF NOT EXISTS idx_txn_sku      ON sales_transactions (sku_id);
CREATE INDEX IF NOT EXISTS idx_txn_channel  ON sales_transactions (channel);
CREATE INDEX IF NOT EXISTS idx_txn_city     ON sales_transactions (city);

-- ── returns ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS returns (
    return_id           VARCHAR(20)     PRIMARY KEY,
    original_txn_id     VARCHAR(20),
    sku_id              VARCHAR(30)     NOT NULL,
    category            VARCHAR(100),
    brand               VARCHAR(100),
    return_date         DATE            NOT NULL,
    quantity_returned   SMALLINT        NOT NULL DEFAULT 1,
    return_reason       VARCHAR(100),
    refund_inr          NUMERIC(10,2),
    channel             VARCHAR(20),
    city                VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_returns_sku  ON returns (sku_id);
CREATE INDEX IF NOT EXISTS idx_returns_date ON returns (return_date);

-- ── supplier_performance ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS supplier_performance (
    id                      BIGSERIAL       PRIMARY KEY,
    supplier_name           VARCHAR(150)    NOT NULL,
    review_month            CHAR(7)         NOT NULL,
    on_time_delivery_pct    NUMERIC(5,2),
    defect_rate_pct         NUMERIC(5,2),
    fill_rate_pct           NUMERIC(5,2),
    lead_time_actual_days   SMALLINT,
    quality_rating          NUMERIC(3,1),
    notes                   TEXT,
    UNIQUE (supplier_name, review_month)
);
CREATE INDEX IF NOT EXISTS idx_sup_perf_supplier ON supplier_performance (supplier_name);

-- ── cold_chain ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cold_chain (
    id                      BIGSERIAL       PRIMARY KEY,
    record_date             DATE            NOT NULL,
    sku_id                  VARCHAR(30)     NOT NULL,
    name                    VARCHAR(255),
    temp_celsius            NUMERIC(5,2),
    target_min_c            NUMERIC(4,2),
    target_max_c            NUMERIC(4,2),
    temp_breach             BOOLEAN         NOT NULL DEFAULT FALSE,
    units_in_cold_storage   SMALLINT,
    expiry_date             DATE,
    shelf_life_days_remaining SMALLINT,
    units_at_risk_of_expiry SMALLINT        DEFAULT 0,
    batch_id                VARCHAR(50)
);
CREATE INDEX IF NOT EXISTS idx_cold_sku   ON cold_chain (sku_id);
CREATE INDEX IF NOT EXISTS idx_cold_date  ON cold_chain (record_date);
CREATE INDEX IF NOT EXISTS idx_cold_breach ON cold_chain (temp_breach);

-- ── store_daily_inventory ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS store_daily_inventory (
    id              BIGSERIAL           PRIMARY KEY,
    record_date     DATE                NOT NULL,
    store_id        VARCHAR(10)         NOT NULL,
    city            VARCHAR(100)        NOT NULL,
    state           VARCHAR(100),
    region          VARCHAR(50)         NOT NULL,
    store_type      VARCHAR(30)         NOT NULL,
    sku_id          VARCHAR(30)         NOT NULL,
    name            VARCHAR(255),
    category        VARCHAR(100)        NOT NULL,
    brand           VARCHAR(100),
    demand          SMALLINT            NOT NULL DEFAULT 0,
    inventory       INT                 NOT NULL DEFAULT 0,
    lead_time_days  SMALLINT            NOT NULL DEFAULT 7,
    days_of_supply  NUMERIC(8,1)        NOT NULL DEFAULT 0,
    risk_status     VARCHAR(10)         NOT NULL DEFAULT 'OK',
    price_inr       NUMERIC(10,2),
    cost_inr        NUMERIC(10,2),
    UNIQUE (record_date, store_id, sku_id)
);
CREATE INDEX IF NOT EXISTS idx_sdi_store    ON store_daily_inventory (store_id);
CREATE INDEX IF NOT EXISTS idx_sdi_date     ON store_daily_inventory (record_date);
CREATE INDEX IF NOT EXISTS idx_sdi_sku      ON store_daily_inventory (sku_id);
CREATE INDEX IF NOT EXISTS idx_sdi_city     ON store_daily_inventory (city);
CREATE INDEX IF NOT EXISTS idx_sdi_region   ON store_daily_inventory (region);
CREATE INDEX IF NOT EXISTS idx_sdi_risk     ON store_daily_inventory (risk_status);
CREATE INDEX IF NOT EXISTS idx_sdi_dos      ON store_daily_inventory (days_of_supply);
CREATE INDEX IF NOT EXISTS idx_sdi_category ON store_daily_inventory (category);

-- ── sku_forecasts (agent writes here) ─────────────────────────────────────
-- forecast_date added to match server.py INSERT and UNIQUE constraint
CREATE TABLE IF NOT EXISTS sku_forecasts (
    id              BIGSERIAL       PRIMARY KEY,
    logged_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    forecast_date   DATE            NOT NULL DEFAULT CURRENT_DATE,
    sku_id          VARCHAR(30)     NOT NULL,
    horizon_days    SMALLINT        NOT NULL DEFAULT 30,
    p10_total       NUMERIC(12,2),
    p50_total       NUMERIC(12,2),
    p90_total       NUMERIC(12,2),
    p50_daily       NUMERIC(10,2),
    forecast_source VARCHAR(100)    NOT NULL DEFAULT 'TFT',
    model_version   VARCHAR(20)     NOT NULL DEFAULT 'v2.0',
    UNIQUE (forecast_date, sku_id, horizon_days, forecast_source)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_sku        ON sku_forecasts (sku_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_date       ON sku_forecasts (forecast_date);
CREATE INDEX IF NOT EXISTS idx_forecasts_logged_at  ON sku_forecasts (logged_at DESC);

-- ── inventory_alerts (agent writes here) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS inventory_alerts (
    id                  BIGSERIAL       PRIMARY KEY,
    alert_date          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    sku_id              VARCHAR(30)     NOT NULL,
    alert_type          VARCHAR(30)     NOT NULL,
    days_of_supply      NUMERIC(8,2),
    current_inventory   INT,
    avg_daily_demand    NUMERIC(10,2),
    lead_time_days      SMALLINT,
    recommended_action  TEXT,
    resolved            BOOLEAN         NOT NULL DEFAULT FALSE,
    resolved_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_alerts_sku      ON inventory_alerts (sku_id);
CREATE INDEX IF NOT EXISTS idx_alerts_date     ON inventory_alerts (alert_date DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON inventory_alerts (resolved);

-- ── demand_anomalies ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS demand_anomalies (
    id              BIGSERIAL       PRIMARY KEY,
    detected_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    sku_id          VARCHAR(30)     NOT NULL,
    anomaly_date    DATE            NOT NULL,
    observed_demand INT             NOT NULL,
    expected_demand NUMERIC(10,2),
    z_score         NUMERIC(6,3),
    anomaly_type    VARCHAR(30),
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_anomalies_sku  ON demand_anomalies (sku_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_date ON demand_anomalies (anomaly_date DESC);

-- ── monthly_kpis ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS monthly_kpis (
    id                  BIGSERIAL       PRIMARY KEY,
    year_month          CHAR(7)         NOT NULL,
    sku_id              VARCHAR(30)     NOT NULL,
    category            VARCHAR(100),
    brand               VARCHAR(100),
    total_demand        BIGINT,
    avg_daily_demand    NUMERIC(10,2),
    avg_inventory       NUMERIC(12,2),
    min_inventory       INT,
    stockout_days       SMALLINT        DEFAULT 0,
    revenue_est_inr     NUMERIC(14,2),
    gross_margin_inr    NUMERIC(14,2),
    fill_rate_pct       NUMERIC(5,2),
    UNIQUE (year_month, sku_id)
);
CREATE INDEX IF NOT EXISTS idx_kpis_sku        ON monthly_kpis (sku_id);
CREATE INDEX IF NOT EXISTS idx_kpis_year_month ON monthly_kpis (year_month);
CREATE INDEX IF NOT EXISTS idx_kpis_category   ON monthly_kpis (category);

-- ── agent_query_log ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_query_log (
    id              BIGSERIAL       PRIMARY KEY,
    session_id      TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
    queried_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provider        VARCHAR(50),
    model           VARCHAR(100),
    user_query      TEXT,
    tools_called    TEXT[],
    duration_ms     INT
);
CREATE INDEX IF NOT EXISTS idx_query_log_date ON agent_query_log (queried_at DESC);

-- ── Views ─────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW forecast_accuracy AS
SELECT sku_id, logged_at, horizon_days, p50_daily, forecast_source
FROM sku_forecasts ORDER BY logged_at DESC;

CREATE OR REPLACE VIEW active_alerts AS
SELECT * FROM inventory_alerts WHERE resolved = FALSE ORDER BY alert_date DESC;
"""

# ══════════════════════════════════════════════════════════════════════════════
#  MySQL migration
# ══════════════════════════════════════════════════════════════════════════════


def migrate_mysql(cfg: dict, label: str) -> None:
    try:
        import pymysql
        from pymysql.constants import CLIENT
    except ImportError:
        print(f"  [SKIP] pymysql not installed. Run: pip install pymysql")
        return

    print(f"\n{'=' * 60}")
    print(f"  MySQL — {label}  ({cfg['host']}:{cfg['port']} / {cfg['database']})")
    print(f"{'=' * 60}")

    try:
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset="utf8mb4",
            client_flag=CLIENT.MULTI_STATEMENTS,
            connect_timeout=10,
        )
    except Exception as e:
        print(f"  [ERROR] Cannot connect: {e}")
        return

    cur = conn.cursor()

    # Drop old tables
    print("  Dropping old tables...")
    for stmt in MYSQL_DROP.strip().split(";"):
        s = stmt.strip()
        if s:
            try:
                cur.execute(s)
            except Exception:
                pass

    # Create new schema — one statement per execute call
    print("  Creating new HUFT schema...")
    for stmt in _MYSQL_TABLES:
        try:
            cur.execute(stmt)
            conn.commit()
        except Exception as e:
            print(f"    [WARN] DDL: {e}")
    conn.commit()

    # Load each CSV
    _mysql_load_csvs(conn, cur, label)
    conn.commit()
    conn.close()
    print(f"  [{label}] MySQL migration complete.")


def _mysql_load_csvs(conn, cur, label: str) -> None:
    import pymysql

    # Column mapping: CSV column name → DB column name (where different)
    LOAD_ORDER = [
        ("products", "products", {}),
        ("stores", "stores", {}),
        ("customers", "customers", {}),
        ("promotions", "promotions", {}),
        ("daily_demand", "daily_demand", {"date": "record_date"}),
        ("sales_transactions", "sales_transactions", {"date": "txn_date"}),
        ("returns", "returns", {}),
        ("supplier_performance", "supplier_performance", {}),
        ("cold_chain", "cold_chain", {"date": "record_date"}),
        ("store_daily_inventory", "store_daily_inventory", {"date": "record_date"}),
    ]

    for csv_key, table, col_map in LOAD_ORDER:
        path = CSV.get(csv_key)
        if path is None or not path.exists():
            print(f"  [SKIP] {csv_key} CSV not found")
            continue

        df = pd.read_csv(path, low_memory=False)
        # Convert ALL NaN / NaT to Python None so pymysql sends SQL NULL
        # (numpy float NaN raises "nan can not be used with MySQL")
        df = df.astype(object).where(pd.notna(df), None)

        # Apply column renames
        if col_map:
            df = df.rename(columns=col_map)

        # Only keep columns that exist in the table
        cur.execute(f"SHOW COLUMNS FROM {table}")
        db_cols = {row[0] for row in cur.fetchall()}
        df = df[[c for c in df.columns if c in db_cols]]

        cols = list(df.columns)
        placeholders = ", ".join(["%s"] * len(cols))
        col_str = ", ".join([f"`{c}`" for c in cols])
        sql = f"INSERT IGNORE INTO {table} ({col_str}) VALUES ({placeholders})"

        rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
        batch = 500
        total = 0
        for i in range(0, len(rows), batch):
            try:
                cur.executemany(sql, rows[i : i + batch])
                conn.commit()
                total += cur.rowcount
            except Exception as e:
                print(f"    [WARN] {table} batch {i // batch}: {e}")

        print(f"  {label} | {table}: {total:,} rows inserted")


# ══════════════════════════════════════════════════════════════════════════════
#  PostgreSQL migration
# ══════════════════════════════════════════════════════════════════════════════


def migrate_postgres(cfg: dict, label: str) -> None:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print(f"  [SKIP] psycopg2 not installed. Run: pip install psycopg2-binary")
        return

    print(f"\n{'=' * 60}")
    print(f"  PostgreSQL — {label}  ({cfg['host']}:{cfg['port']} / {cfg['dbname']})")
    print(f"{'=' * 60}")

    try:
        conn = psycopg2.connect(
            **cfg,
            connect_timeout=30,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=5,
            keepalives_count=5,
        )
        conn.autocommit = False
    except Exception as e:
        print(f"  [ERROR] Cannot connect: {e}")
        return

    cur = conn.cursor()

    # Drop old tables/views
    print("  Dropping old tables...")
    try:
        cur.execute(PG_DROP)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  [WARN] Drop: {e}")

    # Create new schema
    print("  Creating new HUFT schema...")
    try:
        cur.execute(PG_DDL)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  [ERROR] DDL: {e}")
        conn.close()
        return

    # Load each CSV
    _pg_load_csvs(conn, cur, label)
    conn.close()
    print(f"  [{label}] PostgreSQL migration complete.")


def _pg_load_csvs(conn, cur, label: str) -> None:
    import psycopg2.extras

    LOAD_ORDER = [
        ("products", "products", {}),
        ("stores", "stores", {}),
        ("customers", "customers", {}),
        ("promotions", "promotions", {}),
        ("daily_demand", "daily_demand", {"date": "record_date"}),
        ("sales_transactions", "sales_transactions", {"date": "txn_date"}),
        ("returns", "returns", {}),
        ("supplier_performance", "supplier_performance", {}),
        ("cold_chain", "cold_chain", {"date": "record_date"}),
        ("store_daily_inventory", "store_daily_inventory", {"date": "record_date"}),
    ]

    for csv_key, table, col_map in LOAD_ORDER:
        path = CSV.get(csv_key)
        if path is None or not path.exists():
            print(f"  [SKIP] {csv_key} CSV not found")
            continue

        df = pd.read_csv(path, low_memory=False)
        # Convert ALL NaN / NaT to Python None so psycopg2 sends SQL NULL
        df = df.astype(object).where(pd.notna(df), None)

        if col_map:
            df = df.rename(columns=col_map)

        # Get DB columns
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s",
            (table,),
        )
        db_cols = {row[0] for row in cur.fetchall()}
        df = df[[c for c in df.columns if c in db_cols]]

        cols = list(df.columns)
        col_str = ", ".join([f'"{c}"' for c in cols])
        placeholders = ", ".join(["%s"] * len(cols))
        sql = (
            f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholders}) '
            f"ON CONFLICT DO NOTHING"
        )

        rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
        batch = 500
        total = 0
        for i in range(0, len(rows), batch):
            try:
                psycopg2.extras.execute_batch(cur, sql, rows[i : i + batch])
                conn.commit()
                total += len(rows[i : i + batch])
            except Exception as e:
                conn.rollback()
                print(f"    [WARN] {table} batch {i // batch}: {e}")

        print(f"  {label} | {table}: {total:,} rows inserted")

    # ── Populate monthly_kpis from daily_demand ────────────────────────────
    print(f"  {label} | Computing monthly_kpis from daily_demand...")
    try:
        cur.execute("""
            INSERT INTO monthly_kpis
                (year_month, sku_id, category, brand,
                 total_demand, avg_daily_demand, avg_inventory, min_inventory,
                 stockout_days, revenue_est_inr, gross_margin_inr, fill_rate_pct)
            SELECT
                TO_CHAR(record_date, 'YYYY-MM')     AS year_month,
                sku_id,
                MAX(category)                        AS category,
                MAX(brand)                           AS brand,
                SUM(demand)                          AS total_demand,
                ROUND(AVG(demand)::NUMERIC, 2)       AS avg_daily_demand,
                ROUND(AVG(inventory)::NUMERIC, 2)    AS avg_inventory,
                MIN(inventory)                       AS min_inventory,
                SUM(CASE WHEN inventory = 0 THEN 1 ELSE 0 END) AS stockout_days,
                ROUND(SUM(demand * price_inr)::NUMERIC, 2)      AS revenue_est_inr,
                -- gross margin = revenue × margin% (not cost × margin%)
                ROUND(SUM(demand * price_inr * margin_pct / 100)::NUMERIC, 2)
                                                     AS gross_margin_inr,
                ROUND(
                    100.0 * SUM(CASE WHEN inventory > 0 THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0), 2
                )                                    AS fill_rate_pct
            FROM daily_demand
            GROUP BY TO_CHAR(record_date, 'YYYY-MM'), sku_id
            ON CONFLICT (year_month, sku_id) DO NOTHING
        """)
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM monthly_kpis")
        n = cur.fetchone()[0]
        print(f"  {label} | monthly_kpis: {n:,} rows")
    except Exception as e:
        conn.rollback()
        print(f"  [WARN] monthly_kpis: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("\nHUFT Database Migration")
    print("Targets: Local MySQL, Cloud MySQL, Local PostgreSQL, Cloud PostgreSQL")
    print(f"Data directory: {DATA_DIR}\n")

    # Verify CSVs exist
    missing = [k for k, p in CSV.items() if not p.exists()]
    if missing:
        print(f"[ERROR] Missing CSV files: {missing}")
        print("Run:  python data/generate_data.py")
        sys.exit(1)
    print(f"All {len(CSV)} CSV files found.\n")

    # ── MySQL ──────────────────────────────────────────────────────────────
    migrate_mysql(MYSQL_LOCAL, "Local MySQL")
    migrate_mysql(MYSQL_CLOUD, "Cloud MySQL (Railway)")

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    migrate_postgres(PG_LOCAL, "Local PostgreSQL")
    migrate_postgres(PG_CLOUD, "Cloud PostgreSQL (Railway)")

    print("\n" + "=" * 60)
    print("  Migration complete for all 4 databases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
