-- One clean row per sales transaction line. order_id groups the line items of
-- a single basket; customer_id ties the line to a customer. Both are needed by
-- the co_purchase_pairs and customer_product_history marts.
select
    txn_id,
    order_id,
    customer_id,
    date::date            as txn_date,
    sku_id,
    store_id,
    brand,
    category,
    channel,
    customer_segment,
    quantity,
    unit_price_inr,
    discount_pct,
    net_revenue_inr,
    gross_margin_inr
from {{ source('raw', 'transactions') }}
