-- Market-basket recommender (baseline, no ML): for every pair of SKUs bought in
-- the same order, how often they co-occur. This is the "customers who bought X
-- also bought Y" signal, computed from real multi-item baskets (order_id).
-- Downstream: /api/recommendations and the agent's basket-analysis tool.
with lines as (
    select order_id, sku_id
    from {{ ref('stg_transactions') }}
),

-- Self-join within an order; sku_a < sku_b makes each unordered pair appear once.
pairs as (
    select
        a.sku_id as sku_a,
        b.sku_id as sku_b
    from lines a
    join lines b
      on a.order_id = b.order_id
     and a.sku_id < b.sku_id
),

counts as (
    select sku_a, sku_b, count(*) as co_purchases
    from pairs
    group by sku_a, sku_b
),

total as (
    select count(distinct order_id) as total_orders
    from lines
)

select
    c.sku_a,
    pa.name     as sku_a_name,
    pa.category as sku_a_category,
    c.sku_b,
    pb.name     as sku_b_name,
    pb.category as sku_b_category,
    c.co_purchases,
    round(100.0 * c.co_purchases / t.total_orders, 3) as support_pct
from counts c
cross join total t
left join {{ ref('stg_products') }} pa on c.sku_a = pa.sku_id
left join {{ ref('stg_products') }} pb on c.sku_b = pb.sku_id
where c.co_purchases >= 3
order by c.co_purchases desc
