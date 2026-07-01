-- Per-store performance: the numbers the Stores dashboard shows.
with txn as (
    select
        store_id,
        count(distinct txn_id)       as orders,
        sum(quantity)                as units,
        sum(net_revenue_inr)         as revenue,
        sum(gross_margin_inr)        as gross_margin_inr
    from {{ ref('stg_transactions') }}
    group by store_id
),
inv as (
    select
        store_id,
        count(distinct sku_id)                       as sku_count,
        sum(inventory * cost_inr)                    as inventory_value,
        sum(case when risk_status = 'CRITICAL' then 1 else 0 end) as critical_skus,
        sum(case when risk_status = 'WARNING'  then 1 else 0 end) as warning_skus
    from {{ ref('stg_store_inventory') }}
    group by store_id
)
select
    s.store_id,
    s.city,
    s.region,
    s.store_type,
    coalesce(t.orders, 0)            as orders,
    coalesce(t.units, 0)            as units,
    coalesce(t.revenue, 0)          as revenue,
    coalesce(t.gross_margin_inr, 0) as gross_margin_inr,
    case when coalesce(t.revenue, 0) > 0
         then round((100.0 * t.gross_margin_inr / t.revenue)::numeric, 1)
         else 0 end                 as gross_margin_pct,
    coalesce(i.sku_count, 0)        as sku_count,
    coalesce(i.inventory_value, 0)  as inventory_value,
    coalesce(i.critical_skus, 0)    as critical_skus,
    coalesce(i.warning_skus, 0)     as warning_skus
from {{ ref('stg_stores') }} s
left join txn t on s.store_id = t.store_id
left join inv i on s.store_id = i.store_id
