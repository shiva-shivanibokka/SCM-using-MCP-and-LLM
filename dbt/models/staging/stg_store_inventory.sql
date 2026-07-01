-- Latest inventory snapshot per (store, SKU).
select
    store_id,
    sku_id,
    city,
    region,
    category,
    brand,
    demand,
    inventory,
    lead_time_days,
    days_of_supply,
    risk_status,
    price_inr,
    cost_inr
from {{ source('raw', 'store_inventory') }}
