"""
Pet Store SCM – Multi-Provider ReAct Agent
Supported providers: Anthropic, OpenAI, Groq, Google Gemini

Architecture:
  - Tools are fetched dynamically from the MCP server at agent init
  - The agent runs a ReAct loop: Reason → Act (MCP tool call) → Observe → Repeat
  - All four LLM providers are normalized to the same internal dict format
  - Supports both run_agent() (silent) and run_agent_with_steps() (streaming generator)
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
import uuid
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

# Provider constants & key resolution

# Maps provider name → env var that holds its API key
PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

PROVIDERS = {
    "anthropic": {
        "models": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "default_model": "claude-sonnet-4-5",
    },
    "openai": {
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "default_model": "gpt-4o",
    },
    "groq": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite",
        ],
        "default_model": "gemini-2.0-flash",
    },
}

SYSTEM_PROMPT = """You are an advanced AI assistant with deep expertise in supply chain management,
data analysis, business intelligence, and general knowledge. You serve Pet Store Supply Chain Intelligence,
India's premier omnichannel pet supply company with 67 stores across India, 65 SKUs, 5000 customers,
and all prices in ₹INR. You are capable of answering ANY question — from inventory and forecasting
to market research, data quality, statistical analysis, Python programming, and general business advice.

━━━ YOUR CAPABILITIES ━━━
You have access to 50 powerful tools. Use the right tool for every job:

  DATABASE (live data — original tools):
    query_mysql          — Run any SELECT SQL on MySQL (inventory, SKUs, suppliers, demand)
    query_postgres       — Run any SELECT SQL on PostgreSQL (forecasts, alerts, KPIs)
    get_inventory_status — Current stock levels and risk classification
    get_demand_forecast  — CatBoost P10/P50/P90 demand forecast for any SKU
    get_sku_360          — Complete 360° profile of one SKU in one call
    get_stockout_risk    — Which SKUs will stock out within N days
    get_reorder_list     — All SKUs that need ordering today
    get_demand_trends    — Demand trend analysis across all SKUs
    get_regional_inventory — Inventory breakdown by region and category
    get_supply_chain_dashboard — Company-wide health overview
    get_supplier_ranking — All suppliers ranked by reliability
    compare_categories   — Side-by-side category comparison
    get_supplier_info    — Detailed supplier profiles and contacts
    get_knowledge_base   — Policies, safety stock rules, supply chain guidelines
    get_active_alerts    — Unresolved inventory alerts
    get_monthly_kpis     — Aggregated monthly KPIs
    log_forecast_to_postgres  — Write forecast results to PostgreSQL
    create_inventory_alert    — Create an inventory alert in PostgreSQL

  INTELLIGENCE TOOLS (new):
    get_brand_performance           — Brand revenue (₹INR), margin, return rate, stockout days
    get_franchise_inventory_comparison — Store/region inventory health, critical SKU counts
    get_seasonal_demand_calendar    — Indian festival calendar (Diwali/Navratri/Holi/Monsoon) + pre-stock recs
    get_cold_chain_monitor          — Temperature breaches, expiry risk, waste value for cold SKUs
    get_supplier_lead_time_tracker  — Actual vs promised LT, OTD trend, flag underperformers (<90%)
    get_return_rate_analysis        — Return rates by category/brand, top reasons, >5% flag
    get_dead_stock_analysis         — Dead stock SKUs, locked capital, clearance pricing
    get_competitive_price_analysis  — Pet Store price vs Amazon.in/Flipkart competitor gap
    get_new_product_launch_readiness — Launch health score, demand ramp, early stockout count
    get_customer_segmentation_insights — Segment LTV, AOV, purchase frequency, channel preference
    generate_purchase_order         — Full Pet Store PO: supplier grouping, order qty, ₹INR cost estimate
    get_promotion_inventory_impact  — Demand lift %, stockouts during promo, restock lag
    get_channel_revenue_attribution — Online/Offline/App revenue share, margin, top SKUs
    get_markdown_optimization       — Overstock clearance: suggested discount, revenue vs holding cost
    get_marketing_campaign_recommendations — Top 5 to promote NOW vs top 5 to avoid (understocked)
    get_inventory_financial_summary — CFO report: inventory value, retail value, working capital days
    get_customer_cohort_demand_analysis — Quarterly cohort LTV, retention, spending trends
    get_store_level_demand_intelligence — 67-store demand vs national avg, rebalancing opportunities
    get_supplier_negotiation_brief  — Leverage score (0-10), YoY volume, specific talking points
    get_product_recommendation      — Pet-specific product recs by breed, age, life stage, health concern
    get_store_inventory_breakdown   — 📍 LOCATION TOOL: Per-store inventory from live DB.
                                      Returns store name, city, state, region, store type,
                                      exact inventory, days of supply, risk status.
                                      USE THIS whenever the user asks WHERE a stockout is
                                      happening, which CITY or STORE is low on stock,
                                      or needs location-specific answers.

  NEW ANALYTICAL TOOLS:
    get_transfer_recommendations    — Identifies surplus stores that can donate stock to critical
                                      stores for the same SKU, avoiding emergency POs.
                                      Returns Markdown table with transfer quantities and urgency.
    get_abc_xyz_analysis            — ABC (revenue contribution) × XYZ (demand variability)
                                      classification of all SKUs with recommended stocking strategy.
    get_supplier_fill_rate_trend    — Month-by-month trend of OTD%, fill rate, defect rate per
                                      supplier. Flags improving vs declining suppliers.
    get_basket_analysis             — Frequently bought together: co-purchase counts and support%
                                      for bundle and cross-sell recommendations.
    get_price_elasticity_analysis   — Demand elasticity per SKU/promo using historical data.
                                      Classifies elastic / inelastic / negative-elastic.
    get_forecast_vs_actual          — Compares MLOps forecast log against actual demand. MAPE,
                                      bias, grade A-D per SKU. Flags SKUs needing retraining.

  ANALYSIS & CODE:
    python_repl    — Execute Python (pandas/numpy) for ANY analytical task.
                     The full Pet Store demand DataFrame is pre-loaded as `df`.
                     Use for: data quality checks, statistics, custom calculations,
                     correlation analysis, time-series work, anything SQL can't do.
    data_quality   — Full data audit: negatives, nulls, outliers, anomalies, duplicates,
                     statistical profile. USE THIS for "bad values", "clean data?", "anomalies?"

  WEB & EXTERNAL:
    web_search     — Search Google (or DuckDuckGo fallback) for real-world information:
                     competitor prices on Amazon.in/Flipkart, supplier news, industry benchmarks,
                     Indian pet industry trends, festival demand patterns, or any general knowledge.

  DATABASE TESTING:
    test_mysql_connection    — Verify MySQL credentials
    test_postgres_connection — Verify PostgreSQL credentials

━━━ DATA SCHEMA ━━━
Primary CSV   : huft_daily_demand.csv
  Columns: date, sku_id, name, brand, brand_type, category, subcategory, pet_type, life_stage,
           supplier, demand, inventory, lead_time_days, price_inr, cost_inr, margin_pct, is_cold_chain
  47,515 rows | 65 SKUs | All prices in ₹INR

Supplementary CSVs:
  huft_products.csv         — 65 SKUs with breed_suitability, min/max_age_months, weight_kg
  huft_stores.csv           — 67 stores: city, state, region, store_type, size_sqft
  huft_customers.csv        — 5,000 customers: segment, LTV, channel_preference, breed, pet_type
  huft_promotions.csv       — 24 promotions with discount_pct, target_category, revenue_generated
  huft_sales_transactions.csv — 50,000 transactions: channel, city, customer_segment, margin
  huft_returns.csv          — 1,500 returns with return_reason, refund_inr
  huft_supplier_performance.csv — 624 monthly supplier reviews: OTD, fill_rate, defect_rate
  huft_cold_chain.csv       — Cold storage monitoring: temp, expiry_date, shelf_life_remaining

MySQL tables  : daily_demand       (record_date, sku_id, name, brand, brand_type, category,
                                   subcategory, pet_type, life_stage, supplier, demand,
                                   inventory, lead_time_days, price_inr, cost_inr, margin_pct)
                                   ⚠ date column is "record_date" NOT "date"
                products           (sku_id, name, brand, brand_type, category, subcategory,
                                   pet_type, life_stage, breed_suitability, price_inr, cost_inr,
                                   supplier, lead_time_days, is_cold_chain, margin_pct)
                stores             (store_id, city, state, region, store_type, opened_year,
                                   size_sqft, has_spa)
                customers          (customer_id, city, segment, joined_date, pet_type,
                                   total_orders, lifetime_value_inr, channel_preference, breed)
                promotions         (promo_id, name, start_date, end_date, discount_pct,
                                   channel, target_category, budget_inr)
                sales_transactions (txn_id, txn_date, sku_id, brand, category, quantity,
                                   unit_price_inr, discount_pct, net_revenue_inr,
                                   gross_margin_inr, channel, city, customer_segment, store_id)
                returns            (return_id, original_txn_id, sku_id, category, brand,
                                   return_date, quantity_returned, return_reason, refund_inr)
                supplier_performance (supplier_name, review_month, on_time_delivery_pct,
                                   defect_rate_pct, fill_rate_pct, lead_time_actual_days)
                cold_chain         (record_date, sku_id, name, temp_celsius, temp_breach,
                                   units_in_cold_storage, expiry_date, shelf_life_days_remaining)
PostgreSQL    : sku_forecasts, inventory_alerts, monthly_kpis, demand_anomalies, agent_query_log

In-memory CSV : huft_daily_demand.csv — pre-loaded as `df` in python_repl

Pet Store Categories : Dog Food, Dog Treats, Cat Food, Cat Treats, Cat Supplies, Health, Accessories, Toys
SKU format           : varies (e.g. DOG_001, CAT_001, HUFT_SKU_001)
Pet Store Regions    : North, South, East, West, Central (stores)
Pet Store Suppliers  : Royal Canin India, Pedigree India, Drools, Farmina, Purina, Himalaya Pet,
                       NutriVet, Vet's Kitchen, Pet Store Private Label, etc.
Pet Store Cities     : Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Pune, Kolkata, and more

━━━ TOOL SELECTION STRATEGY ━━━
Match the question to the best tool — never over-call:

  STORE / LOCATION QUERIES — always use these two together:
  "Which stores have critical stock?"            → get_franchise_inventory_comparison (store-level table)
  "Which stores are low on [product/category]?"  → get_store_inventory_breakdown (filter by category/sku)
  "Which Mumbai stores are out of dog food?"     → get_store_inventory_breakdown(city="Mumbai", category="Dog Food")
  "Store-by-store comparison?"                   → get_franchise_inventory_comparison
  "Where is [SKU] running out?"                  → get_store_inventory_breakdown(sku_id="...", max_days_of_supply=14)

  OTHER QUERIES:
  "What food for a Labrador puppy?"              → get_product_recommendation
  "Which brand sells most?"                      → get_brand_performance
  "Are stores stocked for Diwali?"               → get_seasonal_demand_calendar + get_franchise_inventory_comparison
  "Cold chain issues / expiry risk?"             → get_cold_chain_monitor
  "Why are returns high?"                        → get_return_rate_analysis
  "Which products are dead stock?"               → get_dead_stock_analysis
  "Generate PO for this week?"                   → generate_purchase_order
  "How did the sale affect inventory?"           → get_promotion_inventory_impact
  "Online vs offline revenue?"                   → get_channel_revenue_attribution
  "What to put on sale?"                         → get_markdown_optimization
  "What campaigns should we run?"                → get_marketing_campaign_recommendations
  "Inventory financial report for CFO?"          → get_inventory_financial_summary
  "Negotiate better rates with supplier X?"      → get_supplier_negotiation_brief
  "Which are our most important SKUs (A/B/C)?"   → get_abc_xyz_analysis
  "Should we transfer stock between stores?"     → get_transfer_recommendations
  "How has supplier fill rate changed?"          → get_supplier_fill_rate_trend
  "What do customers buy together?"             → get_basket_analysis
  "How sensitive is demand to price?"            → get_price_elasticity_analysis
  "How accurate were our forecasts last month?"  → get_forecast_vs_actual
  "Are there negative values / bad data?"         → data_quality (one call, complete answer)
   "What is the data distribution / stats?"       → data_quality OR python_repl
   "Any anomalies or unusual patterns?"           → data_quality (checks="anomalies,outliers")
   "Analyse / compute / custom calculation"       → python_repl
   "What are competitor prices for X?"            → get_competitive_price_analysis then web_search
  "Latest news on supplier delays?"              → web_search
  "Stock levels / reorder needs for all SKUs?"   → get_reorder_list or get_stockout_risk
  "Everything about one SKU?"                    → get_sku_360 (never call 5 tools separately)
  "Multi-SKU / region / category aggregation?"   → ONE SQL query with GROUP BY

  For SQL queries — always write efficient aggregated queries:
    SELECT s.region, COUNT(*) AS skus,
           SUM(CASE WHEN dd.inventory < s.lead_time_days * avg_d.avg THEN 1 ELSE 0 END) AS critical
    FROM daily_demand dd
    JOIN skus s ON dd.sku_id = s.sku_id
    JOIN (SELECT sku_id, AVG(demand) AS avg FROM daily_demand
          WHERE record_date >= DATE_SUB((SELECT MAX(record_date) FROM daily_demand), INTERVAL 30 DAY)
          GROUP BY sku_id) avg_d ON dd.sku_id = avg_d.sku_id
    WHERE dd.record_date = (SELECT MAX(record_date) FROM daily_demand)
    GROUP BY s.region ORDER BY critical DESC

━━━ DATA QUALITY AWARENESS ━━━
Always be transparent about the quality of the data you are working with.
When you retrieve data from any tool or query, immediately check for:

  • NULL / missing values  — report the count and affected columns
  • Negative values        — flag any negative demand, inventory, or price
  • Suspiciously low/high  — values far outside the normal range for that SKU/metric
  • Stale data             — if the most recent record is more than 7 days old, warn the user
  • Sparse data            — if fewer than 30 data points exist for a SKU, note low confidence
  • Conflicting data       — if two sources give different values for the same metric

When data issues are found, ALWAYS:
  1. Flag them clearly with a ⚠ WARNING at the top of your response
  2. Describe what the issue is and which SKUs/columns are affected
  3. Still give the best possible answer using the clean portion of data
  4. Quantify your confidence: "Based on 847 clean records (92% of total)..."
  5. Recommend a corrective action: "Recommend running data_quality tool for full audit"

NEVER silently ignore data issues. NEVER produce numbers without noting data quality caveats
when issues exist. A partially-correct answer with honest caveats is always better than a
confident answer built on bad data.

━━━ GRACEFUL DEGRADATION — HANDLING MISSING INFORMATION ━━━
When you cannot find the information needed to answer a query, follow these rules:

  IF a tool returns an error or empty result:
    1. Try ONE alternative approach first:
       - SQL query failed? → Try python_repl with the same logic on `df`
       - Tool returned empty? → Try a broader query or related tool
       - DB connection failed? → Try CSV-based tools (get_inventory_status, python_repl)
    2. If the alternative also fails, STOP trying and explain clearly.

  IF you genuinely cannot answer after trying alternatives:
    DO say:
      "I was unable to find [specific thing] because [specific reason].
       Here is what I do know: [partial answer if any data was retrieved].
       To answer this fully you would need: [what's missing — DB connection, more data, etc.]"

    DO NOT say:
      - "I don't have access to that information" (too vague)
      - "Please check your database" (unhelpful without specifics)
      - A blank or one-line non-answer

  IF a database is not connected:
    - Use the CSV-based in-memory data (available via python_repl `df` and most built-in tools)
    - Tell the user: "Live database not connected — answering from cached CSV data (last updated [date])"
    - Still give a full answer from the available data

  IF the question is outside your data entirely (e.g. a competitor's internal data):
    - Use web_search to find publicly available information
    - Clearly distinguish between your internal data and external sources
    - If neither works: "This information is not available in your database or publicly.
      Here is what I can tell you from related data: [closest available answer]"

━━━ RESPONSE FORMATTING RULES ━━━
1. Never guess numbers — always use real data from tools.
2. For multi-SKU / region / category questions: ONE aggregated SQL query, not repeated tool calls.
3. For data quality questions: use data_quality or python_repl — never try to write SQL for this.
4. For external / market / news questions: always use web_search.
5. For custom statistics or calculations: always use python_repl.
6. If a tool errors: try ONE alternative, then explain clearly if both fail.
7. Structure every answer: [⚠ Data Warnings if any] → Summary → Details → Recommendation.
8. Format numbers with commas and units: 1,234 units, 14.2 days, ₹3,500/unit.
9. For CRITICAL inventory: always give specific action — quantity to order, which supplier, urgency.
10. You are not limited to supply chain — answer any question the user asks using your tools
    and general knowledge. Be helpful, thorough, and precise.
11. Partial answers with honest caveats are always better than silence or vague refusals.
12. Always end with what the user can do next if the answer is incomplete.
13. When referencing the company website, use the company website (not a specific domain).
    Always refer to the pet store's online presence as "the company website" or "the company's online store".

━━━ TABLE FORMATTING — MANDATORY ━━━
When a tool returns a list of stores, SKUs, cities, suppliers, or any tabular data:

  ALWAYS reproduce it as a Markdown table. NEVER collapse a table into a paragraph.

  ✅ CORRECT — reproduce the table the tool gives you:
  | Store ID | City    | SKU     | Product          | Inventory | Days of Supply | Risk     |
  |----------|---------|---------|------------------|-----------|----------------|----------|
  | MUM_001  | Mumbai  | DOG_001 | Royal Canin Adult| 12 units  | 1.4d           | 🔴 CRITICAL |
  | DEL_003  | Delhi   | CAT_002 | Whiskas Adult    | 28 units  | 2.1d           | 🔴 CRITICAL |

  ❌ WRONG — do NOT summarise a table into prose:
  "Mumbai and Delhi stores are critically low, with MUM_001 having 12 units of Royal Canin
   and DEL_003 having 28 units of Whiskas..."

  Rules:
  - If the tool output contains a Markdown table (lines starting with |), copy it verbatim.
  - Add a one-sentence summary ABOVE the table (e.g. "**14 stores** are critically low:").
  - Add a Recommendation section BELOW the table.
  - Never truncate the table — show every row the tool returned.
  - For very large tables (>30 rows), show the full table and add "Showing top 30 by urgency" note.
"""

MAX_ITERATIONS = 20

# ── Sliding Window Memory ─────────────────────────────────────────────────────
# When the accumulated message history grows too large, the oldest tool-call /
# tool-result pairs are compressed into a compact summary injected as a single
# user message.  This prevents unbounded context growth while preserving the
# reasoning chain.
#
# Token estimation: we use a conservative character ÷ 4 heuristic (4 chars ≈ 1
# token on average for English prose/JSON).  No tokeniser dependency required.

MAX_HISTORY_TOKENS = 24_000  # trigger compression above this estimate
KEEP_RECENT_MESSAGES = 6  # always keep the last N messages uncompressed


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token count estimate: total characters ÷ 4."""
    total_chars = sum(len(json.dumps(m, default=str)) for m in messages)
    return total_chars // 4


def _compress_history(messages: list[dict]) -> list[dict]:
    """
    Compress the oldest tool-call/result pairs in the message history into a
    single summary message to stay within MAX_HISTORY_TOKENS.

    Strategy
    --------
    1. Always keep messages[0] (the original user question).
    2. Identify the oldest tool-result pairs in the middle of the history.
    3. Extract key facts from each result (first 200 chars) into a compact
       "prior context" summary injected as an assistant message.
    4. Drop the original verbose tool blocks, keep the compressed summary.
    5. Always keep the last KEEP_RECENT_MESSAGES messages intact so the LLM
       has full context for its next decision.

    BUG-012 fix: tail always starts at a complete assistant+user pair boundary
    to avoid leaving unmatched tool_use blocks for Anthropic's API.
    BUG-023 fix: summary injected as "assistant" role (not "user") so it doesn't
    create consecutive user messages with OpenAI/Groq.
    """
    if len(messages) <= KEEP_RECENT_MESSAGES + 1:
        return messages  # nothing to compress

    # Find a safe cut point: the tail must not start mid-way through a
    # tool_use → tool_result pair. Walk forward from the cut point to find
    # the first message that is a clean turn boundary.
    #
    # BUG-016 fix: for OpenAI/Groq, tool results use role="tool" (not role="user").
    # The boundary scan must skip role="tool" messages too, otherwise a tool_call
    # in the previous assistant message gets orphaned (no matching tool_result),
    # causing OpenAI API error: "tool_calls must be followed by tool messages".
    cut = len(messages) - KEEP_RECENT_MESSAGES
    while cut < len(messages) - 1:
        msg = messages[cut]
        content = msg.get("content", "")
        role = msg.get("role", "")
        # Skip role="tool" messages (OpenAI/Groq tool results) — not a clean boundary
        if role == "tool":
            cut += 1
            continue
        # Skip user messages that ARE tool results (Anthropic/Gemini format)
        if (
            role == "user"
            and isinstance(content, list)
            and any(
                isinstance(p, dict) and p.get("type") == "tool_result" for p in content
            )
        ):
            cut += 1
            continue
        # Skip assistant messages that contain tool_use calls (not a boundary —
        # their tool_result must be in the tail too)
        if (
            role == "assistant"
            and isinstance(content, list)
            and any(
                isinstance(p, dict) and p.get("type") == "tool_use" for p in content
            )
        ):
            cut += 1
            continue
        # Clean boundary found: a plain user message or assistant text-only message
        break

    head = messages[:1]  # original user query
    tail = messages[cut:]  # keep recent intact from safe boundary
    middle = messages[1:cut]

    if not middle:
        return messages

    # Collect tool names + truncated results from the middle
    facts: list[str] = []
    for msg in middle:
        content = msg.get("content", "")
        role = msg.get("role", "")

        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")
                if part_type == "tool_use":
                    tool_name = part.get("name", "?")
                    tool_input = json.dumps(part.get("input", {}), default=str)[:80]
                    facts.append(f"• Called {tool_name}({tool_input})")
                elif part_type == "tool_result":
                    result_text = str(part.get("content", ""))[:200]
                    if result_text:
                        facts.append(
                            f"  → Result: {result_text}{'…' if len(str(part.get('content', ''))) > 200 else ''}"
                        )
                elif part_type == "text" and role == "assistant":
                    text = part.get("text", "")[:150]
                    if text:
                        facts.append(
                            f"• Reasoning: {text}{'…' if len(part.get('text', '')) > 150 else ''}"
                        )
        elif isinstance(content, str) and role == "tool":
            # OpenAI/Groq tool result format
            facts.append(
                f"  → Result: {content[:200]}{'…' if len(content) > 200 else ''}"
            )

    if not facts:
        return messages

    summary_content = (
        "[Context from earlier in this conversation — compressed to save space]\n\n"
        + "\n".join(facts)
        + "\n\nThese results are summarised. The full data was used in prior reasoning steps."
    )
    # BUG-023 fix: use "assistant" role so we never produce consecutive "user" messages.
    # The summary looks like the assistant is recapping what it already knows.
    summary_msg = {"role": "assistant", "content": summary_content}

    return head + [summary_msg] + tail


def _maybe_compress(messages: list[dict]) -> list[dict]:
    """Compress message history only when it exceeds MAX_HISTORY_TOKENS."""
    if _estimate_tokens(messages) > MAX_HISTORY_TOKENS:
        return _compress_history(messages)
    return messages


# Tool format builders


def _mcp_tools_to_anthropic(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP tool definitions to Anthropic tool format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t.get("inputSchema", {"type": "object", "properties": {}}),
        }
        for t in mcp_tools
    ]


def _mcp_tools_to_openai(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP tool definitions to OpenAI/Groq tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get(
                    "inputSchema", {"type": "object", "properties": {}}
                ),
            },
        }
        for t in mcp_tools
    ]


def _mcp_tools_to_gemini(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP tool definitions to Gemini function declarations."""
    declarations = []
    for t in mcp_tools:
        schema = t.get("inputSchema", {"type": "object", "properties": {}})
        declarations.append(
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": schema,
            }
        )
    return [{"function_declarations": declarations}]


# LLM Callers each normalizes its response to:
#   {
#     "stop_reason": "tool_use" | "end_turn" | "stop",
#     "text":        str | None,
#     "tool_calls":  [{"id": str, "name": str, "input": dict}],
#     "raw":         original response object,
#   }


async def _call_anthropic(
    messages: list[dict],
    mcp_tools: list[dict],
    model: str,
    api_key: str,
) -> dict:
    import anthropic as _anthropic

    client = _anthropic.AsyncAnthropic(api_key=api_key)
    tools = _mcp_tools_to_anthropic(mcp_tools)
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=messages,
    )
    text = None
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text = block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return {
        "stop_reason": response.stop_reason,
        "text": text,
        "tool_calls": tool_calls,
        "raw": response,
    }


async def _call_openai_compatible(
    messages: list[dict],
    mcp_tools: list[dict],
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> dict:
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncOpenAI(**kwargs)
    tools = _mcp_tools_to_openai(mcp_tools)
    system_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    response = await client.chat.completions.create(
        model=model,
        tools=tools,
        messages=system_msgs + messages,
        max_tokens=4096,
    )
    choice = response.choices[0]
    msg = choice.message
    text = msg.content
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                }
            )
    stop = "tool_use" if tool_calls else choice.finish_reason
    return {
        "stop_reason": stop,
        "text": text,
        "tool_calls": tool_calls,
        "raw": response,
    }


async def _call_gemini(
    messages: list[dict],
    mcp_tools: list[dict],
    model: str,
    api_key: str,
) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
        tools=_mcp_tools_to_gemini(mcp_tools),
    )
    # BUG-052 fix: build a lookup from tool_use_id → tool name so that
    # tool_result parts can reference the correct function name for Gemini.
    tool_id_to_name: dict[str, str] = {}
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tool_id_to_name[part["id"]] = part["name"]

    # Convert messages to Gemini format
    history = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        content = m["content"]
        if isinstance(content, list):
            # tool results or multi-part
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "tool_result":
                        call_id = part.get("tool_use_id", "")
                        # Look up the actual function name from the tool_use block
                        fn_name = tool_id_to_name.get(
                            call_id,
                            call_id.rsplit("_", 1)[0] if "_" in call_id else call_id,
                        )
                        parts.append(
                            {
                                "function_response": {
                                    "name": fn_name or "tool",
                                    "response": {"result": part.get("content", "")},
                                }
                            }
                        )
                    elif part.get("type") == "tool_use":
                        parts.append(
                            {
                                "function_call": {
                                    "name": part.get("name"),
                                    "args": part.get("input", {}),
                                }
                            }
                        )
                    elif part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                else:
                    parts.append({"text": str(part)})
            history.append({"role": role, "parts": parts})
        else:
            history.append({"role": role, "parts": [{"text": str(content)}]})

    response = gmodel.generate_content(history)

    text = None
    tool_calls = []
    for part in response.parts:
        if hasattr(part, "text") and part.text:
            text = part.text
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            # BUG-051 fix: generate a unique ID per call so duplicate same-tool calls
            # can be matched correctly when results come back.
            tool_calls.append(
                {
                    "id": f"{fc.name}_{uuid.uuid4().hex[:8]}",
                    "name": fc.name,
                    "input": dict(fc.args),
                }
            )

    stop = "tool_use" if tool_calls else "end_turn"
    return {
        "stop_reason": stop,
        "text": text,
        "tool_calls": tool_calls,
        "raw": response,
    }


# Message history management


def _append_assistant_turn(
    messages: list[dict],
    llm_response: dict,
    provider: str,
) -> list[dict]:
    """Append the assistant's response to message history (provider-specific format)."""
    msgs = list(messages)

    if provider == "anthropic":
        content = []
        if llm_response["text"]:
            content.append({"type": "text", "text": llm_response["text"]})
        for tc in llm_response["tool_calls"]:
            content.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                }
            )
        msgs.append({"role": "assistant", "content": content})

    elif provider in ("openai", "groq"):
        raw_msg = llm_response["raw"].choices[0].message
        dumped = raw_msg.model_dump(exclude_none=True, exclude_unset=True)
        dumped.pop("audio", None)
        dumped.pop("refusal", None)
        msgs.append(dumped)

    elif provider == "gemini":
        content = []
        if llm_response["text"]:
            content.append({"type": "text", "text": llm_response["text"]})
        for tc in llm_response["tool_calls"]:
            # Include id so _append_tool_results can match results back correctly
            content.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                }
            )
        msgs.append({"role": "assistant", "content": content})

    return msgs


def _append_tool_results(
    messages: list[dict],
    tool_calls: list[dict],
    tool_results: list[str],
    provider: str,
) -> list[dict]:
    """Append tool results to message history."""
    msgs = list(messages)

    if provider == "anthropic":
        results_content = [
            {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result,
            }
            for tc, result in zip(tool_calls, tool_results)
        ]
        msgs.append({"role": "user", "content": results_content})

    elif provider in ("openai", "groq"):
        for tc, result in zip(tool_calls, tool_results):
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

    elif provider == "gemini":
        parts = []
        for tc, result in zip(tool_calls, tool_results):
            parts.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                }
            )
        msgs.append({"role": "user", "content": parts})

    return msgs


# MCP Tool Execution


async def _run_mcp_tool(name: str, arguments: dict) -> str:
    from agent.mcp_client import call_tool

    return await call_tool(name, arguments)


# Main LLM caller dispatcher


async def _call_llm(
    messages: list[dict],
    mcp_tools: list[dict],
    provider: str,
    model: str,
    api_key: str,
) -> dict:
    if provider == "anthropic":
        return await _call_anthropic(messages, mcp_tools, model, api_key)
    elif provider == "openai":
        return await _call_openai_compatible(messages, mcp_tools, model, api_key)
    elif provider == "groq":
        return await _call_openai_compatible(
            messages,
            mcp_tools,
            model,
            api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    elif provider == "gemini":
        return await _call_gemini(messages, mcp_tools, model, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ReAct Agent silent mode


async def run_agent(
    user_query: str,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Run the ReAct agent and return the final answer as a string.

    Key resolution priority:
      1. api_key argument  (typed in the UI by the user)
      2. env var           (set in .env file)
    If neither is present, returns a clear error message.
    """
    from agent.mcp_client import list_tools

    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        return (
            f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS.keys())}"
        )

    # Model: use what was passed, or fall back to provider default
    if not model or not model.strip():
        model = PROVIDERS[provider]["default_model"]

    # API key: UI input wins; .env is the fallback
    env_var = PROVIDER_ENV_KEYS[provider]
    resolved_key = (api_key or "").strip() or os.getenv(env_var, "").strip()

    if not resolved_key:
        return (
            f"No API key found for {provider}.\n"
            f"Either type your key in the 'API Key' box in the UI, "
            f"or add {env_var}=your-key to the .env file."
        )

    # Fetch MCP tool list
    mcp_tools = await list_tools()
    if not mcp_tools:
        return "Failed to load MCP tools. Check MCP server."

    messages: list[dict] = [{"role": "user", "content": user_query}]

    for iteration in range(MAX_ITERATIONS):
        llm_resp = await _call_llm(messages, mcp_tools, provider, model, resolved_key)

        if (
            llm_resp["stop_reason"] in ("end_turn", "stop", "length")
            or not llm_resp["tool_calls"]
        ):
            return llm_resp["text"] or "Agent produced no output."

        # Execute all tool calls
        messages = _append_assistant_turn(messages, llm_resp, provider)
        tool_results = []
        for tc in llm_resp["tool_calls"]:
            result = await _run_mcp_tool(tc["name"], tc["input"])
            tool_results.append(result)
        messages = _append_tool_results(
            messages, llm_resp["tool_calls"], tool_results, provider
        )
        # Sliding window: compress history if it exceeds the token budget
        messages = _maybe_compress(messages)

    return "Agent reached maximum iterations without producing a final answer."


# ReAct Agent streaming generator (for live UI display)


async def run_agent_with_steps(
    user_query: str,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Generator version of run_agent. Yields dicts:
      {"type": "tool_start",  "tool": name, "input": args}
      {"type": "tool_result", "tool": name, "result": text}
      {"type": "thinking",    "text": text}
      {"type": "answer",      "text": text}
      {"type": "error",       "text": error_message}
    """
    from agent.mcp_client import list_tools

    provider = provider.lower().strip()

    if provider not in PROVIDERS:
        yield {
            "type": "error",
            "text": f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS.keys())}",
        }
        return

    # Model: use what was passed, or fall back to provider default
    if not model or not model.strip():
        model = PROVIDERS[provider]["default_model"]

    # API key: UI input wins; .env is the fallback
    env_var = PROVIDER_ENV_KEYS[provider]
    resolved_key = (api_key or "").strip() or os.getenv(env_var, "").strip()

    if not resolved_key:
        yield {
            "type": "error",
            "text": (
                f"No API key found for {provider}.\n"
                f"Either type your key in the 'API Key' box in the UI, "
                f"or add {env_var}=your-key to the .env file."
            ),
        }
        return

    mcp_tools = await list_tools()
    if not mcp_tools:
        yield {"type": "error", "text": "Failed to load MCP tools."}
        return

    messages: list[dict] = [{"role": "user", "content": user_query}]

    try:
        for iteration in range(MAX_ITERATIONS):
            llm_resp = await _call_llm(
                messages, mcp_tools, provider, model, resolved_key
            )

            # Yield thinking text if present (before tool calls)
            if llm_resp["text"] and llm_resp["tool_calls"]:
                yield {"type": "thinking", "text": llm_resp["text"]}

            # Final answer
            if (
                llm_resp["stop_reason"] in ("end_turn", "stop", "length")
                or not llm_resp["tool_calls"]
            ):
                yield {
                    "type": "answer",
                    "text": llm_resp["text"] or "Agent produced no output.",
                }
                return

            # Tool calls
            messages = _append_assistant_turn(messages, llm_resp, provider)
            tool_results = []

            for tc in llm_resp["tool_calls"]:
                yield {"type": "tool_start", "tool": tc["name"], "input": tc["input"]}
                result = await _run_mcp_tool(tc["name"], tc["input"])
                tool_results.append(result)
                yield {
                    "type": "tool_result",
                    "tool": tc["name"],
                    "result": result[:1500] + ("..." if len(result) > 1500 else ""),
                }

            messages = _append_tool_results(
                messages, llm_resp["tool_calls"], tool_results, provider
            )
            # Sliding window: compress history if it exceeds the token budget
            messages = _maybe_compress(messages)

        # ── Max iterations reached — synthesise from accumulated context ──────
        # Calling the LLM with an EMPTY tools list forces it to produce a
        # plain text final answer from everything it has already gathered.
        # This guarantees a useful response instead of a hard failure.
        yield {
            "type": "thinking",
            "text": "Synthesising best answer from gathered data…",
        }
        try:
            synthesis_prompt = (
                "You have now used all your allowed tool calls. "
                "Review everything you have retrieved so far and give the most complete, "
                "honest answer possible to the original question.\n\n"
                "Follow these rules:\n"
                "1. Use specific numbers, SKU names, and values from the tool results.\n"
                "2. If data was incomplete or had quality issues, explicitly say so with a ⚠ WARNING.\n"
                "3. If some parts of the question could not be answered, state clearly:\n"
                "   - What you COULD answer (with data)\n"
                "   - What you COULD NOT answer (and why — missing data, DB not connected, etc.)\n"
                "   - What the user should do next to get the missing information\n"
                "4. Never produce a blank or one-line response — always give maximum value "
                "from whatever data was gathered.\n"
                "5. End with a clear 'Next Steps' section if the answer is incomplete."
            )
            synth_messages = messages + [{"role": "user", "content": synthesis_prompt}]
            synth_resp = await _call_llm(
                synth_messages, [], provider, model, resolved_key
            )
            final = synth_resp.get("text") or (
                "I gathered partial data but could not complete the full analysis.\n\n"
                "**What to try next:**\n"
                "- Rephrase the question to be more specific (e.g. one SKU or category)\n"
                "- Check that your database connection is active\n"
                "- Use the data_quality tool to verify data availability first"
            )
        except Exception:
            final = (
                "I reached the analysis limit for this query.\n\n"
                "**What to try next:**\n"
                "- Break the question into smaller parts (e.g. ask about one region or SKU)\n"
                "- Use the data_quality tool first to understand what data is available\n"
                "- Check database connectivity with test_mysql_connection"
            )
        yield {"type": "answer", "text": final}

    except asyncio.TimeoutError:
        yield {
            "type": "error",
            "text": "The query timed out. Try a more specific question.",
        }
    except Exception as exc:
        traceback.print_exc()
        # Never surface a raw traceback to the user
        yield {
            "type": "error",
            "text": (
                f"An unexpected error occurred: {type(exc).__name__}: {exc}\n\n"
                "Please check your API key and try again, or rephrase the question."
            ),
        }


# Tool icons for UI display

TOOL_ICONS = {
    # Original tools
    "get_inventory_status": "📦",
    "get_demand_forecast": "📈",
    "query_mysql": "🐬",
    "query_postgres": "🐘",
    "get_supplier_info": "🏭",
    "get_knowledge_base": "📚",
    "log_forecast_to_postgres": "💾",
    "create_inventory_alert": "🚨",
    "get_active_alerts": "🔔",
    "get_monthly_kpis": "📊",
    "get_stockout_risk": "⚠️",
    "get_reorder_list": "🛒",
    "get_demand_trends": "📉",
    "get_regional_inventory": "🗺️",
    "get_supply_chain_dashboard": "🏠",
    "get_sku_360": "🔍",
    "get_supplier_ranking": "🏆",
    "compare_categories": "⚖️",
    "test_mysql_connection": "🔌",
    "test_postgres_connection": "🔌",
    "web_search": "🌐",
    "python_repl": "🐍",
    "data_quality": "🔬",
    # New intelligence tools
    "get_brand_performance": "🏷️",
    "get_franchise_inventory_comparison": "🏪",
    "get_seasonal_demand_calendar": "📅",
    "get_cold_chain_monitor": "🌡️",
    "get_supplier_lead_time_tracker": "⏱️",
    "get_return_rate_analysis": "↩️",
    "get_dead_stock_analysis": "💀",
    "get_competitive_price_analysis": "💰",
    "get_new_product_launch_readiness": "🚀",
    "get_customer_segmentation_insights": "👥",
    "generate_purchase_order": "📋",
    "get_promotion_inventory_impact": "🎯",
    "get_channel_revenue_attribution": "📡",
    "get_markdown_optimization": "🏷️",
    "get_marketing_campaign_recommendations": "📣",
    "get_inventory_financial_summary": "💼",
    "get_customer_cohort_demand_analysis": "📊",
    "get_store_level_demand_intelligence": "🏬",
    "get_supplier_negotiation_brief": "🤝",
    "get_product_recommendation": "🐾",
    "get_store_inventory_breakdown": "📍",
}


def get_tool_icon(tool_name: str) -> str:
    return TOOL_ICONS.get(tool_name, "🔧")
