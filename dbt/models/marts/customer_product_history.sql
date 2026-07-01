-- Per-customer purchase history: one row per (customer, SKU) with how often and
-- how much they've bought it. This is the personalisation signal — the input a
-- collaborative-filtering recommender would train on, and enough on its own to
-- power "your usual" / reorder suggestions without any model.
select
    t.customer_id,
    t.sku_id,
    p.name              as sku_name,
    p.category          as category,
    count(*)            as times_purchased,
    sum(t.quantity)     as units,
    sum(t.net_revenue_inr) as revenue,
    max(t.txn_date)     as last_purchase_date
from {{ ref('stg_transactions') }} t
left join {{ ref('stg_products') }} p on t.sku_id = p.sku_id
group by t.customer_id, t.sku_id, p.name, p.category
