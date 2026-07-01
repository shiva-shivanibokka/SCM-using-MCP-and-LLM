-- Supplier reliability, averaged across all review months — one row per supplier.
select
    supplier_name,
    count(*)                                       as review_months,
    round(avg(on_time_delivery_pct)::numeric, 1)   as on_time_delivery_pct,
    round(avg(defect_rate_pct)::numeric, 2)        as defect_rate_pct,
    round(avg(fill_rate_pct)::numeric, 1)          as fill_rate_pct,
    round(avg(lead_time_actual_days)::numeric, 1)  as lead_time_days,
    round(avg(quality_rating)::numeric, 2)         as quality_rating
from {{ ref('stg_suppliers') }}
group by supplier_name
