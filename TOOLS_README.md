# Pet Store Supply Chain Intelligence — Complete Tool Reference

**AI Agent — 50 Tools in Plain English**

This document explains every tool the AI agent has access to, what it does, when to use it, and example questions you can ask.

---

## How to Use This

Just type your question naturally into the chat. The agent automatically picks the right tool. You don't need to know which tool to use — just ask what you want to know.

---

## Section 1: Inventory & Stock Tools

### 1. `get_inventory_status`
**What it does:** Shows you the current stock level of any product and whether it's running low.

**When to use it:** When you want a quick check on whether a product is in stock, running out, or already out.

**Example questions:**
- *"Is Royal Canin Labrador food in stock?"*
- *"Which products are about to run out?"*
- *"Show me the top 10 most at-risk SKUs right now"*

---

### 2. `get_stockout_risk`
**What it does:** Predicts which products will run out of stock within the next N days based on how fast they're selling.

**When to use it:** For planning ahead — before a sale, festival, or busy season.

**Example questions:**
- *"Which products will stock out in the next 7 days?"*
- *"What needs urgent reordering before Diwali?"*

---

### 3. `get_reorder_list`
**What it does:** Gives you a ready-made list of every product that needs to be reordered today, sorted by urgency.

**When to use it:** Your daily procurement checklist.

**Example questions:**
- *"What do I need to reorder today?"*
- *"Give me today's reorder list"*

---

### 4. `get_sku_360`
**What it does:** Gives a complete profile of one product — stock level, sales trend, supplier info, safety stock, reorder point, and recommended order quantity — all in one answer.

**When to use it:** When you need everything about a specific product.

**Example questions:**
- *"Tell me everything about Sara's Chicken & Rice fresh food"*
- *"Full profile of FOOD_D001"*

---

### 5. `get_supply_chain_dashboard`
**What it does:** A company-wide health check — how many products are critical, how many are fine, overall inventory health score.

**When to use it:** Morning briefing, weekly review, management reports.

**Example questions:**
- *"Give me the overall supply chain health report"*
- *"How is our inventory doing company-wide?"*

---

### 6. `get_regional_inventory`
**What it does:** Breaks down inventory levels by region (North/South/East/West) or product category.

**When to use it:** When you want to know which part of India has stock problems.

**Example questions:**
- *"How is dog food inventory in South India?"*
- *"Which region has the most critical stock situations?"*

---

### 7. `get_demand_forecast`
**What it does:** Predicts how much demand a specific product will have over the next 14–90 days, with a pessimistic, expected, and optimistic estimate.

**When to use it:** Before placing a large order or before a promotion.

**Example questions:**
- *"Forecast demand for Pedigree Puppy food for the next 30 days"*
- *"How much Royal Canin will we sell in the next 2 weeks?"*

---

### 8. `get_demand_trends`
**What it does:** Shows which products' demand is growing, stable, or declining over the past 90 days.

**When to use it:** For spotting trends before they become problems.

**Example questions:**
- *"Which products have growing demand this quarter?"*
- *"Are any categories declining in sales?"*

---

## Section 2: Brand & Product Tools

### 9. `get_brand_performance`
**What it does:** Shows how each brand is performing — total revenue, units sold, profit margin, return rate, and how often products from that brand go out of stock.

**When to use it:** For brand reviews, annual negotiations, deciding which brands to stock more or less of.

**Example questions:**
- *"How is Royal Canin performing vs Drools?"*
- *"Which brand has the highest return rate?"*
- *"Show me the top 10 brands by revenue"*
- *"Detailed performance report for Farmina"*

---

### 10. `get_new_product_launch_readiness`
**What it does:** When you launch a new product, this checks if it's properly stocked across all channels, how fast demand ramped up in the first month, and gives a health score (0–100).

**When to use it:** After launching a new SKU or private label product.

**Example questions:**
- *"How is the new Sara's Mutton fresh food launch going?"*
- *"Is FOOD_D012 properly stocked after launch?"*

---

### 11. `get_product_recommendation`
**What it does:** Recommends the right products for a specific pet — based on pet type, breed, age, and any health concerns. Answers questions from pet parents about what to buy.

**When to use it:** Customer-facing queries, staff training, product selection guidance.

**Example questions:**
- *"What food is best for a 4-month-old Labrador puppy?"*
- *"What should I feed a 2-year-old Golden Retriever?"*
- *"My cat has a tick problem — what products do you recommend?"*
- *"What treats are good for a senior Beagle?"*
- *"My German Shepherd puppy is 3 months old — what food do you sell for her?"*

---

## Section 3: Supplier Tools

### 12. `get_supplier_info`
**What it does:** Detailed profile of any supplier — contact info, on-time delivery rate, quality rating, minimum order quantity, emergency capability.

**Example questions:**
- *"Tell me about Royal Canin India as a supplier"*
- *"Which suppliers can handle emergency orders?"*

---

### 13. `get_supplier_ranking`
**What it does:** Ranks all suppliers from best to worst based on reliability, lead times, and quality.

**Example questions:**
- *"Who are our most reliable suppliers?"*
- *"Which supplier is causing the most delays?"*

---

### 14. `get_supplier_lead_time_tracker`
**What it does:** Tracks whether suppliers are delivering on time — shows actual delivery time vs what was promised, trends over the last 6 months, and flags anyone falling below 90% on-time delivery.

**When to use it:** Before re-negotiating contracts or placing large orders.

**Example questions:**
- *"Is Boehringer Ingelheim delivering NexGard on time?"*
- *"Show me lead time performance for all suppliers over the last 6 months"*
- *"Which suppliers are getting worse at delivery?"*

---

### 15. `get_supplier_negotiation_brief`
**What it does:** Prepares a negotiation brief for any supplier — how much business we have grown with them, current terms, leverage score (0–10), and specific talking points for the next negotiation meeting.

**When to use it:** Before supplier meetings or annual contract renewals.

**Example questions:**
- *"Prepare a negotiation brief for Mars Petcare India"*
- *"Which suppliers should we renegotiate with this year?"*
- *"Give me leverage analysis for Drools"*

---

## Section 4: Cold Chain & Quality Tools

### 16. `get_cold_chain_monitor`
**What it does:** Monitors Sara's fresh food products specifically — checks for temperature breaches, products close to expiry, and estimates waste value.

**When to use it:** Daily check for fresh food operations. Critical for Sara's Wholesome range.

**Example questions:**
- *"Any temperature issues with Sara's fresh food today?"*
- *"How much fresh food is at risk of expiry in the next 7 days?"*
- *"Cold chain health report"*

---

### 17. `get_return_rate_analysis`
**What it does:** Shows how often products are being returned, why customers return them, and flags any product with a return rate above 5% (the industry warning level).

**When to use it:** For quality monitoring and identifying product issues early.

**Example questions:**
- *"Which products have the highest return rates?"*
- *"Why are customers returning dog food?"*
- *"Is Farmina's return rate within acceptable limits?"*

---

### 18. `get_dead_stock_analysis`
**What it does:** Finds products that haven't sold in 60+ days, calculates how much money is locked up in unsold stock, and recommends what discount is needed to clear it within 30 days.

**When to use it:** Before the end of a season, before a new collection arrives, or any time you want to free up warehouse space and cash.

**Example questions:**
- *"Which products are sitting unsold?"*
- *"How much capital is locked in dead stock?"*
- *"What discount do I need to give to clear slow-moving cat trees?"*

---

## Section 5: Marketing Tools

### 19. `get_seasonal_demand_calendar`
**What it does:** Shows which products and categories see demand spikes in which months, aligned with Indian festivals (Diwali, Navratri, Holi, Dussehra, Independence Day) and seasons (monsoon for tick/flea, summer for cooling products, winter for clothing).

**When to use it:** For planning inventory pre-stocking and marketing campaigns 2–3 months ahead.

**Example questions:**
- *"What should we stock up on before Diwali?"*
- *"When does tick and flea product demand peak?"*
- *"Plan inventory for the next 3 months"*
- *"What categories spike during monsoon season?"*

---

### 20. `get_promotion_inventory_impact`
**What it does:** Measures what happens to inventory during and after a promotion — how much demand went up, which products stocked out during the sale, and how long it took to restock.

**When to use it:** After any sale or campaign, to learn what to do differently next time.

**Example questions:**
- *"What was the inventory impact of the Diwali 2024 sale?"*
- *"Which products ran out during our last promotion?"*
- *"Did the Republic Day offer cause any stockout problems?"*

---

### 21. `get_competitive_price_analysis`
**What it does:** Searches Google and competitor websites to find prices for the same products on Amazon.in, Flipkart, and other platforms. Compares them to our prices and tells you if you're priced competitively.

**When to use it:** Before changing prices, during sale planning, or when a customer says "I found it cheaper elsewhere."

**Example questions:**
- *"Is our price for Royal Canin competitive vs Amazon?"*
- *"Are we priced right on Pedigree Puppy food?"*
- *"Where are we overpriced compared to competitors?"*

---

### 22. `get_marketing_campaign_recommendations`
**What it does:** Tells your marketing team which product categories to promote right now (because they're overstocked or about to hit a seasonal peak) and which categories NOT to promote (because stock is too low to handle extra demand).

**When to use it:** Weekly or monthly marketing planning.

**Example questions:**
- *"What categories should we run a campaign on this week?"*
- *"What should our marketing team promote next month?"*
- *"Which products should we NOT advertise right now because we can't fulfil demand?"*

---

### 23. `get_markdown_optimization`
**What it does:** Finds products that are overstocked and calculates the exact discount percentage needed to sell through them within 30 days. Compares discount revenue vs the cost of continuing to hold the stock.

**When to use it:** Planning the "Offer Zone 60% Off" section or any clearance sale.

**Example questions:**
- *"What should go into next month's clearance sale and at what discount?"*
- *"Which dog beds are overstocked and by how much?"*
- *"Optimize markdowns for the accessories category"*

---

## Section 6: Customer & Channel Tools

### 24. `get_customer_segmentation_insights`
**What it does:** Breaks down our 5,000 customers into segments (New Pet Parent, Loyal Premium, Budget Conscious, etc.) and shows what each segment buys, how often, through which channel, and their lifetime value.

**When to use it:** For targeted marketing, personalised promotions, and understanding who your most valuable customers are.

**Example questions:**
- *"Who are our most valuable customers?"*
- *"What do Loyal Premium customers buy most?"*
- *"Which customer segment prefers online shopping?"*
- *"How does the New Pet Parent segment differ from Multi-Pet Households?"*

---

### 25. `get_channel_revenue_attribution`
**What it does:** Breaks down revenue, units sold, and profit margin by sales channel — Online (website), Offline (stores), and App.

**When to use it:** For understanding which channel is growing, which is most profitable, and where to focus investment.

**Example questions:**
- *"How much revenue comes from online vs offline stores?"*
- *"Which channel has the best profit margin?"*
- *"Is the app growing as a sales channel?"*
- *"What are the top-selling products online vs in stores?"*

---

### 26. `get_customer_cohort_demand_analysis`
**What it does:** Groups customers by when they joined and tracks their buying behaviour over time — how much they spend in month 1 vs month 6, how many stay loyal, and what products they progress through as their pet ages.

**When to use it:** For understanding customer lifetime value and planning what to stock as your customer base ages with their pets.

**Example questions:**
- *"Do customers who joined in 2023 spend more than those who joined in 2022?"*
- *"What's our customer retention rate at 6 months?"*
- *"What do new customers buy vs customers who've been with us for a year?"*

---

### 27. `get_store_level_demand_intelligence`
**What it does:** Compares demand patterns across all 67 stores. Shows which stores have unique buying patterns vs the national average, which are chronically understocked, and where rebalancing stock between stores could save money.

**When to use it:** For regional supply chain decisions and store-specific inventory allocation.

**Example questions:**
- *"Which stores are consistently understocked on cat food?"*
- *"Does the Koramangala Bengaluru store have different demand than the national average?"*
- *"Where can we rebalance excess stock from one store to another?"*

---

## Section 7: Financial Tools

### 28. `get_inventory_financial_summary`
**What it does:** A CFO-level financial report on inventory — total value of stock on hand, how much could be made if all stock sells, how much is locked in dead stock, estimated revenue lost from stockouts, and working capital days.

**When to use it:** Monthly financial reviews, board presentations, investor reports.

**Example questions:**
- *"What is the total value of our current inventory?"*
- *"How much revenue are we losing from stockouts?"*
- *"How many days of working capital is tied up in inventory?"*
- *"Give me the inventory financial health report"*

---

### 29. `generate_purchase_order`
**What it does:** Automatically generates a purchase order for all products that need restocking — grouped by supplier, with quantities, estimated costs in INR, and expected delivery dates. Can filter by urgency (critical only, or all reorders).

**When to use it:** When you want to turn the reorder list into an actual purchase order ready to send to suppliers.

**Example questions:**
- *"Generate a purchase order for all critical items"*
- *"Create a PO for everything that needs restocking from Royal Canin India"*
- *"What should we order this week and what will it cost?"*

---

## Section 8: Franchise Tools

### 30. `get_franchise_inventory_comparison`
**What it does:** Compares inventory health across all 120 franchise stores — which stores are dangerously low on stock, which are overstocked, and which regions have the worst supply situation.

**When to use it:** For franchise operations management and deciding which stores get priority replenishment.

**Example questions:**
- *"Which franchise stores are most at-risk right now?"*
- *"Compare inventory health across North vs South India stores"*
- *"Which stores will run out of Royal Canin in the next 7 days?"*
- *"Show me inventory comparison for all West India stores"*

---

## Section 9: Category & Comparison Tools

### 31. `compare_categories`
**What it does:** Side-by-side comparison of all product categories — Dog Food vs Cat Food vs Health vs Toys vs Accessories — showing health scores, risk levels, and top urgent products in each.

**Example questions:**
- *"Which product category is in the worst shape right now?"*
- *"Compare all categories"*

---

## Section 10: Data & Analytics Tools

### 32. `python_repl`
**What it does:** Runs Python code directly on the store's data. Used for custom calculations, statistical analysis, or any question that needs flexible data crunching.

**Example questions:**
- *"What is the correlation between price and return rate?"*
- *"Show me a statistical summary of demand for all dog food SKUs"*
- *"Are there any negative inventory values in our data?"*

---

### 33. `data_quality`
**What it does:** Runs a full health check on the data — finds negative values, missing data, statistical outliers, demand spikes, and duplicates.

**Example questions:**
- *"Is there any bad data in our system?"*
- *"Are there any unnatural values in inventory or demand?"*
- *"Run a data quality check"*

---

### 34. `web_search`
**What it does:** Searches Google for any external information — competitor prices, industry news, supplier disruptions, market trends, or any general knowledge question.

**Example questions:**
- *"What are competitors charging for Royal Canin on Amazon?"*
- *"Are there any supply chain disruptions affecting pet food imports from Europe?"*
- *"What is the market size of the pet food industry in India in 2024?"*

---

## Section 11: Database Tools

### 35. `query_mysql`
**What it does:** Run any custom SQL query directly on the live MySQL database (inventory, orders, products, suppliers).

**Example questions:**
- *"How many units of Pedigree did we sell last month?"*
- *"Show me all reorder events in the last 30 days"*

---

### 36. `query_postgres`
**What it does:** Run any custom SQL query on the PostgreSQL analytics database (forecasts, alerts, KPIs).

**Example questions:**
- *"Show me all active inventory alerts"*
- *"What are the monthly KPIs for Q3 2024?"*

---

### 37. `get_knowledge_base`
**What it does:** Access the internal knowledge base — reorder policies, safety stock guidelines, cold chain rules, supplier terms.

**Example questions:**
- *"What is the reorder policy?"*
- *"What is the safety stock formula used?"*

---

### 38–43. Connection & Logging Tools

| Tool | What it does |
|---|---|
| `test_mysql_connection` | Checks if MySQL is connected and working |
| `test_postgres_connection` | Checks if PostgreSQL is connected and working |
| `log_forecast_to_postgres` | Saves a forecast result to the database |
| `create_inventory_alert` | Creates a new inventory alert in the system |
| `get_active_alerts` | Shows all unresolved inventory alerts |
| `get_monthly_kpis` | Fetches monthly KPI aggregates |

---

---

## Section 9: Advanced Analytics Tools

### 45. `get_transfer_recommendations`
**What it does:** Identifies overstocked stores that can send surplus inventory to critically understocked stores for the same SKU — avoiding expensive emergency purchase orders.

**Example questions:**
- *"Should we move stock between stores to cover the critical ones?"*
- *"Which stores have surplus they can donate to Mumbai?"*

---

### 46. `get_abc_xyz_analysis`
**What it does:** Classifies every SKU by revenue contribution (A/B/C) and demand variability (X/Y/Z). AX SKUs must always be in stock; CZ SKUs should be considered for discontinuation.

**Example questions:**
- *"Which are our most important SKUs?"*
- *"Give me the ABC analysis for dog food"*
- *"Which products have erratic demand?"*

---

### 47. `get_supplier_fill_rate_trend`
**What it does:** Shows how each supplier's on-time delivery %, fill rate, and defect rate have changed month by month. Flags suppliers getting worse vs improving.

**Example questions:**
- *"Which suppliers are getting worse at delivery over time?"*
- *"How has Royal Canin India's fill rate trended over 6 months?"*

---

### 48. `get_basket_analysis`
**What it does:** Finds which products are most frequently bought together in the same order. Used for bundle recommendations, shelf placement, and cross-sell campaigns.

**Example questions:**
- *"What do customers buy together with dog food?"*
- *"Which products should we bundle for a promotion?"*
- *"Give me the top cross-sell opportunities"*

---

### 49. `get_price_elasticity_analysis`
**What it does:** Uses historical promotion data to estimate how much demand changes when you apply a discount. Classifies products as elastic (responds well to discounts), inelastic (price doesn't matter), or negative-elastic (discounting hurts premium perception).

**Example questions:**
- *"Which products respond most to discounts?"*
- *"Should we run a sale on cat food?"*
- *"Which SKUs are price sensitive?"*

---

### 50. `get_forecast_vs_actual`
**What it does:** Compares what the forecasting model predicted against what actually sold. Shows MAPE (% error), bias (over/under forecast), and grades each SKU A–D. Grade D SKUs are flagged for retraining.

**Example questions:**
- *"How accurate were our forecasts last month?"*
- *"Which products have the worst forecast error?"*
- *"Which SKUs need model retraining?"*

---

## Quick Reference: What Question → What Tool

| If you want to know... | Use this tool |
|---|---|
| Is a product in stock? | `get_inventory_status` |
| What will stock out soon? | `get_stockout_risk` |
| What to order today? | `get_reorder_list` |
| Everything about one product? | `get_sku_360` |
| What food for my puppy? | `get_product_recommendation` |
| How is a brand performing? | `get_brand_performance` |
| Which supplier is unreliable? | `get_supplier_lead_time_tracker` |
| Are cold-chain products safe? | `get_cold_chain_monitor` |
| What to stock before Diwali? | `get_seasonal_demand_calendar` |
| Which products to put on sale? | `get_dead_stock_analysis` or `get_markdown_optimization` |
| What to promote this week? | `get_marketing_campaign_recommendations` |
| Revenue by online vs offline? | `get_channel_revenue_attribution` |
| How much inventory is worth? | `get_inventory_financial_summary` |
| Generate a purchase order? | `generate_purchase_order` |
| Which stores need restocking? | `get_franchise_inventory_comparison` |
| Are there data quality issues? | `data_quality` |
| What are competitors charging? | `get_competitive_price_analysis` |
| Prepare for supplier meeting? | `get_supplier_negotiation_brief` |

---

| Should we transfer stock between stores? | `get_transfer_recommendations` |
| Which SKUs are our most important (A/B/C)? | `get_abc_xyz_analysis` |
| Which supplier is getting worse over time? | `get_supplier_fill_rate_trend` |
| What do customers buy together? | `get_basket_analysis` |
| Which products respond best to discounts? | `get_price_elasticity_analysis` |
| How accurate were last month's forecasts? | `get_forecast_vs_actual` |

---

*Last updated: April 2026 | Tools: 50 | Pet Store Supply Chain Intelligence Platform*
