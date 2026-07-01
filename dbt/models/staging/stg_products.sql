-- One clean row per SKU.
select
    sku_id,
    name,
    brand,
    brand_type,
    category,
    subcategory,
    pet_type,
    life_stage,
    price_inr,
    cost_inr,
    margin_pct,
    supplier,
    lead_time_days,
    is_cold_chain
from {{ source('raw', 'products') }}
