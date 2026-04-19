-- ============================================================
-- Pet Store Supply Chain – MySQL Schema
-- Database: pet_store_scm
-- ============================================================

CREATE DATABASE IF NOT EXISTS pet_store_scm
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE pet_store_scm;

-- ------------------------------------------------------------
-- 1. SKU master (static product catalog)
-- ------------------------------------------------------------
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
    INDEX idx_category  (category),
    INDEX idx_supplier  (supplier),
    INDEX idx_region    (region)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 2. Suppliers master
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_name       VARCHAR(150)    PRIMARY KEY,
    contact_email       VARCHAR(255),
    contact_phone       VARCHAR(50),
    country             VARCHAR(100),
    on_time_delivery_pct DECIMAL(5,2),   -- e.g. 96.20
    quality_rating      DECIMAL(3,1),    -- 1.0-5.0
    min_order_qty       SMALLINT UNSIGNED,
    emergency_capable   BOOLEAN DEFAULT FALSE,
    notes               TEXT,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 3. Daily demand + inventory fact table  (MySQL owns this)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_demand (
    id              BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
    record_date     DATE              NOT NULL,
    sku_id          VARCHAR(20)       NOT NULL,
    demand          MEDIUMINT UNSIGNED NOT NULL DEFAULT 0,
    inventory       MEDIUMINT UNSIGNED NOT NULL DEFAULT 0,
    lead_time_days  TINYINT UNSIGNED  NOT NULL,
    price_usd       DECIMAL(10,2)     NOT NULL,
    UNIQUE KEY uq_date_sku (record_date, sku_id),
    INDEX idx_sku       (sku_id),
    INDEX idx_date      (record_date),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id) ON UPDATE CASCADE
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 4. Reorder events log (MySQL owns purchasing decisions)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reorder_events (
    id              BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
    event_date      DATE              NOT NULL,
    sku_id          VARCHAR(20)       NOT NULL,
    trigger_reason  ENUM('LOW_STOCK','SCHEDULED','EMERGENCY','FORECAST') NOT NULL,
    ordered_qty     MEDIUMINT UNSIGNED NOT NULL,
    expected_arrival DATE,
    actual_arrival   DATE,
    unit_cost       DECIMAL(10,2),
    status          ENUM('PENDING','IN_TRANSIT','RECEIVED','CANCELLED') NOT NULL DEFAULT 'PENDING',
    created_at      TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sku_date (sku_id, event_date),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id) ON UPDATE CASCADE
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- 5. Supplier performance log
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS supplier_performance (
    id              BIGINT UNSIGNED   AUTO_INCREMENT PRIMARY KEY,
    supplier_name   VARCHAR(150)      NOT NULL,
    review_date     DATE              NOT NULL,
    on_time_pct     DECIMAL(5,2),
    defect_rate_pct DECIMAL(5,2),
    fill_rate_pct   DECIMAL(5,2),
    notes           TEXT,
    UNIQUE KEY uq_supplier_date (supplier_name, review_date),
    FOREIGN KEY (supplier_name) REFERENCES suppliers(supplier_name) ON UPDATE CASCADE
) ENGINE=InnoDB;

-- ============================================================
-- Seed Suppliers
-- ============================================================
INSERT IGNORE INTO suppliers (supplier_name, contact_email, country, on_time_delivery_pct, quality_rating, min_order_qty, emergency_capable, notes) VALUES
('PawsSupply Co',    'orders@pawssupply.com',    'United States', 96.20, 4.7, 50,  TRUE,  'Primary distributor. Same-day emergency orders up to 500 units.'),
('NutriPet Inc',     'supply@nutripetin.com',    'United States', 92.50, 4.5, 100, FALSE, 'Specialty nutrition supplier. MOQ 100 units.'),
('GlobalPet Dist',   'intl@globalpetdist.com',   'Mexico',        88.10, 4.1, 200, FALSE, 'Latin America distributor. Longer lead times; customs delays possible.'),
('TreatWorld LLC',   'b2b@treatworld.com',        'China',         84.30, 3.9, 500, FALSE, 'Asia Pacific treats supplier. Quality flag raised Q4 2023, resolved Q1 2024.'),
('HealthPet Labs',   'orders@healthpetlabs.com', 'United States', 94.80, 4.6, 50,  TRUE,  'Pharmaceutical-grade supplements. FDA registered facility.'),
('VetPharma Supply', 'vet@vetpharma.eu',         'Germany',       91.20, 4.8, 30,  FALSE, 'EU-based vet pharma distributor. Strong quality, longer shipping.'),
('PetEssentials',    'wholesale@petessentials.com','United States',97.10, 4.4, 100, TRUE,  'General accessories distributor. Largest catalog breadth.'),
('ToyPet Factory',   'orders@toypetfactory.com', 'China',         80.50, 3.7, 1000,FALSE, 'Budget toy supplier. High MOQ. Lead time variance ±5 days.'),
('RawPet Foods',     'cold@rawpetfoods.com',     'United States', 95.00, 4.9, 20,  TRUE,  'Premium raw/freeze-dried specialist. Cold chain logistics.');
