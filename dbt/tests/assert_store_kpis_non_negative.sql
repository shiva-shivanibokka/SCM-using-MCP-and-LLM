-- Singular data-quality test: revenue and inventory value must never be negative.
-- dbt passes when this query returns zero rows.
select store_id, revenue, inventory_value
from {{ ref('store_kpis') }}
where revenue < 0
   or inventory_value < 0
