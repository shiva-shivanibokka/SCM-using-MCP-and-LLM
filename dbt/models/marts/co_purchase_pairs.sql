-- Market-basket recommender (baseline, no ML): for every pair of SKUs bought in
-- the same order, how often they co-occur AND how strongly they're associated.
--
-- Raw co-purchase counts alone just rank by popularity (a popular item pairs with
-- everything), so we also compute:
--   confidence A→B = P(B | A) = co_purchases / orders_containing_A
--   lift           = P(A,B) / (P(A)·P(B)) — >1 means bought together more than
--                    chance would predict (genuine affinity), ~1 means incidental.
-- Per-product recommendations rank by lift so "customers who bought X also bought
-- Y" surfaces real complements, not just whatever else is popular.
with lines as (
    select order_id, sku_id
    from {{ ref('stg_transactions') }}
),

pairs as (
    select a.sku_id as sku_a, b.sku_id as sku_b
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

sku_orders as (   -- how many orders contain each SKU
    select sku_id, count(distinct order_id) as orders
    from lines
    group by sku_id
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
    round(100.0 * c.co_purchases / t.total_orders, 3)                    as support_pct,
    round((c.co_purchases * t.total_orders)::numeric / (oa.orders * ob.orders), 2) as lift,
    round(100.0 * c.co_purchases::numeric / oa.orders, 1)               as conf_a_to_b,
    round(100.0 * c.co_purchases::numeric / ob.orders, 1)               as conf_b_to_a
from counts c
cross join total t
join sku_orders oa on c.sku_a = oa.sku_id
join sku_orders ob on c.sku_b = ob.sku_id
left join {{ ref('stg_products') }} pa on c.sku_a = pa.sku_id
left join {{ ref('stg_products') }} pb on c.sku_b = pb.sku_id
where c.co_purchases >= 3
order by c.co_purchases desc
