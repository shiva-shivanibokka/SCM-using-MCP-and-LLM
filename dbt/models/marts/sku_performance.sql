-- Per-SKU sales performance, enriched with product attributes.
with sales as (
    select
        sku_id,
        sum(quantity)         as units,
        sum(net_revenue_inr)  as revenue,
        sum(gross_margin_inr) as gross_margin_inr
    from {{ ref('stg_transactions') }}
    group by sku_id
)
select
    p.sku_id,
    p.name,
    p.brand,
    p.category,
    p.pet_type,
    coalesce(s.units, 0)            as units,
    coalesce(s.revenue, 0)         as revenue,
    coalesce(s.gross_margin_inr, 0) as gross_margin_inr,
    case when coalesce(s.revenue, 0) > 0
         then round((100.0 * s.gross_margin_inr / s.revenue)::numeric, 1)
         else 0 end                as gross_margin_pct
from {{ ref('stg_products') }} p
left join sales s on p.sku_id = s.sku_id
