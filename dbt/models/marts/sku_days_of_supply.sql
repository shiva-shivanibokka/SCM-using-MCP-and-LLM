-- Per-SKU inventory coverage, rolled up across stores from the latest snapshot.
-- Reads the raw store_inventory directly (rather than stg) so it can take the
-- most recent row per (store, SKU) whether the source is the Postgres compact
-- snapshot or the full DuckDB/Parquet daily history.
with inv as (
    select
        store_id,
        sku_id,
        demand,
        inventory,
        days_of_supply,
        risk_status,
        date
    from {{ source('raw', 'store_inventory') }}
),

ranked as (
    select
        inv.*,
        row_number() over (
            partition by store_id, sku_id order by date desc
        ) as rn
    from inv
),

latest as (
    select * from ranked where rn = 1
)

select
    sku_id,
    count(distinct store_id)                                      as stores_stocked,
    round(avg(days_of_supply), 1)                                 as avg_days_of_supply,
    sum(inventory)                                                as total_inventory,
    sum(demand)                                                   as total_daily_demand,
    sum(case when risk_status = 'CRITICAL' then 1 else 0 end)     as critical_stores,
    sum(case when risk_status = 'WARNING'  then 1 else 0 end)     as warning_stores
from latest
group by sku_id
