-- Monthly supplier scorecard rows (one per supplier per review month).
select
    supplier_name,
    review_month,
    on_time_delivery_pct,
    defect_rate_pct,
    fill_rate_pct,
    lead_time_actual_days,
    quality_rating
from {{ source('raw', 'suppliers') }}
