-- One clean row per store.
select
    store_id,
    city,
    state,
    region,
    store_type,
    opened_year,
    size_sqft,
    has_spa
from {{ source('raw', 'stores') }}
