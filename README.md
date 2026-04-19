# Pet Store Supply Chain Intelligence Platform
### Model Context Protocol · Temporal Fusion Transformer · Multi-LLM ReAct Agent · Real-Time Analytics

A production-grade, end-to-end **AI-powered supply chain intelligence system** built for a premium Indian pet retail company with 67 stores across India. The system combines a **44-tool MCP server**, a **multi-provider ReAct agent**, a **Temporal Fusion Transformer (TFT)** demand forecasting model, and a **17-chart interactive analytics dashboard** — all served through a polished Gradio web application.

---

## What Makes This Different

Most supply chain tools are dashboards that show you what happened. This system reasons over your data, calls live database tools, searches the web, executes Python code, and tells you what to do — in natural language.

- **Not RAG** — No vector databases, no embeddings, no similarity search. Instead, the agent uses **Model Context Protocol (MCP)** to call typed tools that query live MySQL and PostgreSQL databases in real time.
- **Not a chatbot** — The agent follows a full **ReAct reasoning loop** (Reasoning → Acting → Observing) with up to 20 tool calls per query, synthesising data from multiple sources before answering.
- **Not a single model** — The forecasting engine uses a **Temporal Fusion Transformer** (state-of-the-art for retail time-series) with automatic fallback to **CatBoost**, both enriched with promotion calendar and Indian festival features.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Gradio Web Application (5 Tabs)                      │
│  AI Assistant │ Inventory Dashboard │ Analytics │ Demand Forecast │ MLOps│
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              ReAct Agent  (agent/agent.py)                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐ │
│  │Anthropic │  │  OpenAI  │  │   Groq   │  │   Google Gemini          │ │
│  │ Claude   │  │  GPT-4o  │  │ LLaMA 3  │  │   Flash / Pro            │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────────────┘ │
│  Hot-swappable at runtime · Session-scoped API keys · Auto model lists   │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │  MCP Protocol (in-process or SSE/HTTP)
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  MCP Server  (mcp_server/server.py)                      │
│                  FastAPI + uvicorn · SSE + JSON-RPC 2.0                  │
│                  50 Tools  ·  4 Resources  ·  /health  ·  /docs          │
└──────┬────────────────┬───────────────────────┬──────────────────────────┘
       │                │                       │
       ▼                ▼                       ▼
  MySQL DB        PostgreSQL DB          In-Memory / CSV
  (inventory,     (forecasts,            (HUFT datasets,
   suppliers,      alerts, KPIs,          promotion calendar,
   orders)         audit log)             cold chain data)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            Forecasting Engine  (forecasting/)                            │
│  ┌─────────────────────────────────────┐  ┌──────────────────────────┐  │
│  │  Temporal Fusion Transformer (TFT) │  │  CatBoost Fallback       │  │
│  │  pytorch-forecasting 1.7.0         │  │  3 quantile models       │  │
 │  │  GPU/CPU · FP16 mixed prec.        │  │  (P10 / P50 / P90)       │  │
│  │  Full retrain: ~30 min             │  │  Trains in ~2 min        │  │
│  │  Fine-tune:    ~5 min              │  │  Always available        │  │
│  └─────────────────────────────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Highlights

### 1. Model Context Protocol (MCP) — 50 Tools

The agent does not have hardcoded SQL or logic. It calls **50 typed MCP tools** that the LLM selects dynamically based on the user's query. The tools are grouped into seven categories:

| Category | Tools | What They Do |
|---|---|---|
| **Core Database** | `query_mysql`, `query_postgres`, `get_inventory_status`, `get_sku_360`, `get_stockout_risk`, `get_reorder_list`, `get_demand_trends`, `get_regional_inventory`, `get_supply_chain_dashboard`, `get_demand_forecast` | Live SQL queries + computed inventory metrics |
| **Supplier & Knowledge** | `get_supplier_info`, `get_supplier_ranking`, `get_knowledge_base`, `get_active_alerts`, `get_monthly_kpis`, `compare_categories`, `test_mysql_connection`, `test_postgres_connection` | Supplier intelligence + policies + alert management |
| **Business Intelligence** | `get_brand_performance`, `get_franchise_inventory_comparison`, `get_store_inventory_breakdown`, `get_seasonal_demand_calendar`, `get_cold_chain_monitor`, `get_supplier_lead_time_tracker`, `get_return_rate_analysis`, `get_dead_stock_analysis`, `get_new_product_launch_readiness`, `get_competitive_price_analysis` | Operational intelligence with per-store location tables |
| **Marketing Intelligence** | `get_promotion_inventory_impact`, `get_channel_revenue_attribution`, `get_markdown_optimization`, `get_marketing_campaign_recommendations`, `get_customer_segmentation_insights`, `get_customer_cohort_demand_analysis` | Marketing ↔ inventory linkage |
| **Financial & Strategic** | `get_inventory_financial_summary`, `generate_purchase_order`, `get_store_level_demand_intelligence`, `get_supplier_negotiation_brief`, `get_product_recommendation` | C-suite level reporting + automated PO generation |
| **Advanced Analytics** | `get_transfer_recommendations`, `get_abc_xyz_analysis`, `get_supplier_fill_rate_trend`, `get_basket_analysis`, `get_price_elasticity_analysis`, `get_forecast_vs_actual` | Stock transfers, ABC-XYZ classification, basket analysis, price elasticity, forecast accuracy |
| **AI Capabilities** | `web_search`, `python_repl`, `data_quality` | Google Search (SerpAPI + DuckDuckGo fallback), sandboxed Python execution, full data audit |

**MCP Transport:** The server runs as a FastAPI application with SSE and JSON-RPC 2.0. By default it runs **in-process** (no separate server needed). Set `BYPASS_MCP_HTTP=false` to use real HTTP communication for distributed deployments.

---

### 2. Multi-Provider ReAct Agent — Architecture Deep Dive

The agent is built from three layers: an **LLM layer**, an **orchestration layer**, and a **tool layer**. Each is independently replaceable.

---

#### Layer 1 — LLM (Reasoning Engine)

The agent supports **four LLM providers** hot-swappable at runtime without restarting the app:

| Provider | Models | Notes |
|---|---|---|
| **Anthropic** | claude-opus-4-5, claude-sonnet-4-5, claude-3-5-haiku | Recommended — best multi-step reasoning |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | Strong general performance |
| **Groq** (free tier) | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b | Very fast inference |
| **Google Gemini** | gemini-2.0-flash, gemini-1.5-pro, gemini-2.0-flash-lite | Good long-context handling |

All four providers are normalized to the same internal response format `{stop_reason, text, tool_calls, raw}` by a provider-specific adapter. The orchestration loop never knows which provider it's talking to.

**How each provider receives the system prompt:**
- **Anthropic** — `system=SYSTEM_PROMPT` as a top-level parameter (not in the messages array)
- **OpenAI / Groq** — injected as `messages[0] = {"role":"system","content":SYSTEM_PROMPT}`
- **Gemini** — `system_instruction=SYSTEM_PROMPT` on the `GenerativeModel()` constructor

---

#### Layer 2 — Orchestration (ReAct Loop)

The `run_agent_with_steps()` function in `agent/agent.py` implements the full **Reason → Act → Observe** loop with a maximum of **20 iterations**:

```
User query
   │
   ▼  ─── up to 20 iterations ──────────────────────────────────────────┐
_call_llm(messages, tools, provider)                                     │
   │                                                                     │
   ├─ stop_reason == "end_turn"  OR  no tool_calls                       │
   │    └─ yield {"type":"answer"} ──► shown in chat bubble  DONE        │
   │                                                                     │
   └─ tool_calls present                                                 │
        ├─ yield {"type":"thinking"} ──► shown as quoted reasoning text  │
        ├─ for each tool_call:                                           │
        │    ├─ yield {"type":"tool_start"}  (silently dropped by UI)    │
        │    ├─ await _run_mcp_tool(name, args) ──► MCP dispatch         │
        │    └─ yield {"type":"tool_result"} (silently dropped by UI)    │
        ├─ _append_assistant_turn(messages)  ← update history            │
        ├─ _append_tool_results(messages)    ← update history            │
        └─ _maybe_compress(messages)         ← sliding window ──────────┘
   │
   ▼  if 20 iterations reached without a final answer:
_call_llm(messages, tools=[])   ← empty tools list FORCES plain text
yield {"type":"answer"}         ← synthesis from all gathered data
```

**Key orchestration properties:**
- **Live streaming** — `run_agent_with_steps` is an `AsyncGenerator`. Each `yield` triggers a Gradio UI update so users see reasoning steps as they happen
- **Async-to-sync bridge** — Gradio is synchronous; a daemon Thread + Queue bridges the async generator to Gradio's sync generator
- **Forced synthesis** — if the agent hits 20 iterations, the LLM is called with `tools=[]` which forces it to produce a plain-text answer from all data gathered so far, instead of a hard failure
- **Graceful degradation** — every tool call is wrapped in try/except that returns a structured error message; the LLM can read the error and try an alternative approach

**Sliding Window Memory** (added to prevent unbounded context growth):

Token usage grows with every tool call because the full result is appended to the message history. For long multi-step queries (10+ tool calls), this can hit context limits. The sliding window compresses old tool-result pairs into a compact summary:

```
Before compression (iteration 12):
  [user query] [tool_use: get_inventory_status] [result: 2,000 chars]
  [tool_use: query_mysql] [result: 3,000 chars] ... × 10 more pairs

After _maybe_compress() triggers at 24,000 token estimate:
  [user query]
  [SUMMARY: • Called get_inventory_status() → DOG_001: 42 units, CRITICAL...
             • Called query_mysql(SELECT...) → 5 rows returned...
             • ... 8 more compressed entries]
  [last 6 messages kept intact]  ← recent context preserved fully
```

The compression is lossless for decision-making: key facts (tool names, truncated results) are preserved, verbose raw data is trimmed. This keeps the context window bounded regardless of query complexity.

---

#### Layer 3 — Tools (MCP Server)

**50 tools** served via `mcp_server/server.py`. The tool layer has three sub-components:

**3a. Tool Registry** (new — replaces static list + if/elif):

Previously, tools were registered as a 400-line static `MCP_TOOLS` list and dispatched via a 50-branch `if/elif` chain. The new architecture uses a `_TOOL_REGISTRY` dict populated at module load time:

```python
# Every non-DB tool is now registered with its schema, handler, and caching config
_TOOL_REGISTRY["get_inventory_status"] = {
    "schema":    MCP_TOOLS entry (description, inputSchema),
    "handler":   tool_get_inventory_status,
    "cacheable": True,     # pure CSV read — safe to cache
    "cache_ttl": 60,       # seconds
}
```

`dispatch_tool()` routes registered tools through `_call_registered_tool()` (fast dict lookup + cache check) and only falls through to the legacy `_dispatch_tool_inner()` for the 9 DB tools that need credential injection.

**3b. Tool Result Caching** (new):

Pure read-only CSV tools are marked `cacheable=True`. Identical calls within 60 seconds return the cached result instantly:

```
First call: get_inventory_status(sku_id="DOG_001")
  → runs pandas computation (~50ms) → caches result for 60s

Second call (25s later): get_inventory_status(sku_id="DOG_001")
  → cache hit → returns in <1ms
```

Cache is keyed on `(tool_name, md5(sorted_args_json))`. The 9 DB tools (MySQL/PostgreSQL reads, writes, alerts) are never cached — they always hit the live database.

Tool cache can be invalidated explicitly: `invalidate_tool_cache("get_inventory_status")` or `invalidate_tool_cache()` for all.

**3c. Transport**

```
BYPASS_MCP_HTTP=true (default):
  agent → call_tool_direct() → dispatch_tool() → function call
  In-process, zero network latency.

BYPASS_MCP_HTTP=false (HTTP mode for distributed deployments):
  agent → POST /messages (JSON-RPC 2.0) → FastAPI → dispatch_tool()
  Wire format: {"jsonrpc":"2.0","method":"tools/call","params":{"name":...,"arguments":{...}}}
```

The `/sse` endpoint (`GET /sse`) is an MCP spec-compliant SSE handshake channel but all actual tool call data flows through `POST /messages`.

**Tool categories:**

| Category | Count | Transport | Cacheable |
|---|---|---|---|
| Pure CSV / pandas | ~30 | Direct call | Yes (60s TTL) |
| MySQL reads | 1 (`query_mysql`) | aiomysql async | No |
| PostgreSQL reads/writes | 6 | asyncpg async | No |
| Hybrid (tries MySQL→PG→CSV) | 1 (`get_store_inventory_breakdown`) | async | No |
| External (web, python REPL) | 2 | subprocess/HTTP | No |

---

#### Message History Format

The history is a `list[dict]` that grows over the ReAct loop. Format differs by provider:

**Anthropic / Gemini (multi-part content blocks):**
```python
# After LLM decides to call a tool:
{"role": "assistant", "content": [
    {"type": "text",     "text": "I'll check the inventory levels first."},
    {"type": "tool_use", "id": "call_abc123", "name": "get_inventory_status", "input": {...}}
]}
# After tool result comes back:
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_abc123", "content": "DOG_001: 42 units..."}
]}
```

**OpenAI / Groq (raw SDK message objects):**
```python
# After LLM decides to call a tool (raw SDK dump):
{"role": "assistant", "tool_calls": [
    {"id": "call_xyz", "type": "function", "function": {"name": "get_inventory_status", "arguments": "{...}"}}
]}
# After tool result:
{"role": "tool", "tool_call_id": "call_xyz", "content": "DOG_001: 42 units..."}
```

The sliding window compressor understands both formats and compresses them uniformly.

---

#### What Connects the Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Gradio UI (gradio_app.py)                     │
│  chat() generator                                                    │
│    ├─ set_session_creds()  ──► ContextVar (per-request DB creds)    │
│    └─ Thread(asyncio.run(_collect()))                                │
│         └─ run_agent_with_steps()  [agent/agent.py]                 │
│              ├─ list_tools()  ──────► _TOOL_REGISTRY schemas        │
│              └─ ReAct Loop                                           │
│                   ├─ _call_llm() ──► Anthropic/OpenAI/Groq/Gemini   │
│                   ├─ yield thinking/answer to Gradio live            │
│                   ├─ _run_mcp_tool() ──► mcp_client.call_tool()     │
│                   └─ _maybe_compress() ──► sliding window           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ BYPASS_MCP_HTTP=true (default)
                                 ▼
              ┌──────────────────────────────────────────────────────┐
              │          mcp_server/server.py                         │
              │  dispatch_tool()                                       │
              │    ├─ _TOOL_REGISTRY tools  ──► _call_registered_tool │
              │    │    └─ TTL cache check ──► tool function()        │
              │    └─ DB tools  ──► _dispatch_tool_inner()            │
              │         └─ ContextVar creds ──► aiomysql / asyncpg   │
              └──────────────────────────────────────────────────────┘
```

The agent follows a full **ReAct loop** (up to 20 iterations) with:
- Live streaming of the agent's reasoning process in the UI
- Transparent tool call display with emoji icons per tool
- Structured graceful degradation — if a tool fails, the agent tries an alternative and explains what went wrong
- Data quality awareness — the agent flags anomalies, nulls, and low-confidence data inline with every answer
- Sliding window memory — compresses old tool results to prevent context overflow on long queries
- Tool result caching — identical CSV tool calls within 60 seconds return instantly from cache
- 300-second timeout for complex multi-step queries

---

### 3. Temporal Fusion Transformer (TFT) Forecasting

The forecasting engine uses a **state-of-the-art Temporal Fusion Transformer** — the same architecture used by major retailers for demand planning — with a CatBoost fallback for zero-downtime availability.

**What TFT understands that simple models don't:**

| Feature Type | Examples | Why It Matters for HUFT |
|---|---|---|
| **Known future covariates** | Diwali date, promotion schedule, monsoon season | Model sees the festival is coming and pre-adjusts the forecast |
| **Time-varying unknowns** | Past demand, rolling averages, lag features | Captures weekly/seasonal patterns |
| **Static categoricals** | SKU ID, brand, category, breed suitability, cold chain flag | Each SKU gets its own learned embedding |
| **Static reals** | Price, lead time, margin %, base demand | Encodes product-level characteristics |

**Promotion calendar features (11 new features):**
`is_promo_active`, `promo_discount_pct`, `days_to_next_promo`, `is_festival_week`, `is_diwali_season`, `is_navratri`, `is_monsoon`, `is_summer`, `is_winter`, `is_holi`, `is_independence_day`

**GPU optimisations (works on any CUDA GPU; tested on RTX 4060 8GB):**
- FP16 mixed precision (`precision="16-mixed"`) — 40% less VRAM; falls back to CPU automatically if no CUDA GPU
- Gradient accumulation × 4 — effective batch size 256 without OOM
- `gradient_clip_val=0.1` — essential for TFT stability
- `hidden_size=128, heads=4, dropout=0.1` — tuned for ≤8GB VRAM
- `MAX_ENCODER=90` days — fixed memory per sample, scales to millions of rows
- `num_workers=0` — required on Windows

**MLOps Monitor — three training modes:**
- **Full TFT Retrain (~30 min on GPU)** — trains from scratch on all history
- **Fine-tune TFT (~5 min)** — loads checkpoint, updates on last 90 days
- **CatBoost Fallback (~2 min)** — always available, runs on CPU

**Metrics reported:** MAPE, MAE, RMSE, P10–P90 calibration %

---

### 4. HUFT Dataset — 9 Synthetic Datasets Modelled on Real Business

All data is synthetically generated to match HUFT's actual product catalog, store network, supplier base, and Indian market dynamics.

| Dataset | Rows | Description |
|---|---|---|
| `huft_daily_demand.csv` | 47,515 | Daily demand + inventory for 65 SKUs × 730 days (2023–2024) |
| `huft_sales_transactions.csv` | 50,000 | Individual transactions with channel, city, segment, margin |
| `huft_customers.csv` | 5,000 | Customer records with segment, breed, LTV, channel preference |
| `huft_products.csv` | 65 | Full product master — brand, breed suitability, life stage, cold chain flag |
| `huft_stores.csv` | 67 | All stores — 60 physical across India + online + spa locations |
| `huft_promotions.csv` | 24 | HUFT promotions — Diwali, Navratri, Holi, Monsoon Drive, Republic Day |
| `huft_returns.csv` | 1,500 | Return log with reasons (~3% return rate) |
| `huft_supplier_performance.csv` | 624 | Monthly supplier scorecards for 26 HUFT suppliers |
| `huft_cold_chain.csv` | 2,193 | Temperature monitoring + expiry tracking for cold-chain SKUs (freeze-dried and raw food lines) |

**65 real HUFT SKUs across 15 categories:**
Royal Canin, Pedigree, Farmina, Drools, Whiskas, Temptations, NexGard, Frontline, Bravecto, Virbac, KONG, Trixie, Ruffwear, Wahl, Cats Best — plus HUFT private labels: **Sara's Wholesome** (fresh food, cold chain), **Hearty**, **Meowsi**, **Dash Dog**, **HUFT branded** accessories, grooming, toys, clothing, bedding.

**India-specific demand seasonality:**
Diwali (+45% demand spike), Navratri, Dussehra, Holi, Independence Day, Republic Day, monsoon season (tick/flea products), summer (cooling mats, grooming), winter (dog clothing, bedding) — all encoded in the TFT's known future covariates.

---

### 5. Interactive Analytics Dashboard — 17 Switchable Charts

A fully dynamic **5-tab Gradio application** where every chart is controlled by dropdowns and updates instantly without reloading the page.

#### Tab 1 — Inventory Dashboard (5 switchable views)
| View | Chart Type | What It Shows |
|---|---|---|
| Inventory Health Heatmap | Horizontal bar heatmap | All SKUs coloured red/amber/green by days of supply |
| Days of Supply | Sorted bar chart | Days of supply per SKU with CRITICAL/WARNING/OK threshold lines |
| Inventory vs Demand | Grouped bar | Current inventory vs 30-day demand side by side |
| Stockout Risk Timeline | Gantt-style bar | Which SKUs stock out and when (next 30 days) |
| Dead Stock Analysis | Stacked bar | Capital locked in dead/slow/active stock per category (₹INR) |

#### Tab 2 — Analytics Dashboard (Marketing + Operational + Management)

**Marketing Analytics** (6 switchable charts):
| Chart | Description |
|---|---|
| Sales by Channel | Stacked area — Online vs Offline vs App revenue over time |
| Brand Performance | Bubble chart — Revenue × Margin % × Return rate per brand |
| Category Revenue Heatmap | 12-month × N-category heatmap showing seasonal peaks |
| Promotion Impact | Before/during/after demand comparison per campaign |
| Top SKUs by Revenue | Horizontal bar, filterable by category and channel |
| Customer Segments | Donut chart — Revenue contribution per customer segment |

**Operational Analytics** (4 switchable charts):
| Chart | Description |
|---|---|
| Lead Time Performance | Scatter — Actual vs promised lead time per supplier (diagonal = perfect) |
| Cold Chain Monitor | Multi-line temperature trend with danger zone shading |
| Seasonal Demand Index | Radar/polar chart — 12-month demand index per category |
| Reorder Events Timeline | Monthly stacked bar — purchasing activity by category |

**Management Dashboard** (3 switchable charts + always-visible KPI cards):
| Chart | Description |
|---|---|
| Financial KPI Cards | Total inventory value, retail value, dead stock, lost revenue, working capital days — all in ₹INR |
| Private Label vs Third Party | Donut — HUFT brands vs third-party revenue split |
| Month-over-Month Growth | Indexed line chart — revenue, units, gross margin over 24 months |
| Store Inventory Comparison | Horizontal bar — all 67 stores ranked by inventory health score |

#### Tab 3 — Demand Forecast
Per-SKU probabilistic forecast with P10/P50/P90 fan chart, 90-day inventory history with risk zone shading, reorder point and safety stock lines, and quantile recommendations.

#### Tab 4 — MLOps Monitor
Model training (Full TFT / Fine-tune / CatBoost), forecast accuracy by SKU (MAPE, MAE, RMSE, calibration), drift detection, agent query log.

---

### 6. Advanced Agent Capabilities

Beyond supply chain queries, the agent handles:

- **Web search** — Google (SerpAPI) with DuckDuckGo fallback: competitor prices on Amazon.in, supplier news, industry benchmarks, regulatory updates
- **Python REPL** — Sandboxed execution with pandas + numpy pre-loaded and the full HUFT DataFrame as `df`. Dangerous imports (`os`, `sys`, `subprocess`) blocked at AST level
- **Data quality audit** — One-call full audit: negatives, nulls, per-SKU z-score outliers (|z|>3), demand spike anomalies (>3× rolling mean), inventory drop anomalies (>80% single day), duplicate detection
- **Product recommendations** — Breed + age + health-specific product matching from HUFT's catalog. *"What food suits a 4-month-old Labrador puppy?"* returns breed-specific recommendations with price and availability
- **Purchase order generation** — Automatically generates a formatted PO grouped by supplier with quantities, cost in ₹INR, and expected delivery dates
- **Supplier negotiation briefs** — Leverage score (0–10) and talking points based on YoY volume growth, OTD trends, and fill rates

---

## Project Structure

```
SCM-using-MCP-and-LLM/
│
├── gradio_app.py                  ← Main app (5 tabs, 17 charts, full UI)
├── requirements.txt
├── TOOLS_README.md                ← Plain-English guide to all 50 tools
│
├── agent/
│   ├── agent.py                   ← ReAct agent, 4 LLM providers, SYSTEM_PROMPT
│   └── mcp_client.py              ← MCP client (in-process or SSE/HTTP)
│
├── mcp_server/
│   └── server.py                  ← MCP server: 50 tools, FastAPI, SSE+JSON-RPC
│
├── forecasting/
│   ├── ml_forecast.py             ← TFT primary + CatBoost fallback
│   └── data_loader.py             ← Multi-source loader (MySQL/PostgreSQL/CSV)
│
├── data/
│   ├── generate_data.py           ← HUFT synthetic data generator (9 CSVs)
│   ├── huft_daily_demand.csv      ← 47,515 rows · 65 SKUs · 730 days
│   ├── huft_sales_transactions.csv← 50,000 transactions
│   ├── huft_customers.csv         ← 5,000 customers
│   ├── huft_products.csv          ← 65 products (real HUFT brands)
│   ├── huft_stores.csv            ← 67 stores across India
│   ├── huft_promotions.csv        ← 24 HUFT promotions
│   ├── huft_returns.csv           ← 1,500 returns
│   ├── huft_supplier_performance.csv ← 624 supplier scorecards
│   └── huft_cold_chain.csv        ← 2,193 cold chain records
│
├── db/
│   ├── mysql_schema.sql           ← MySQL schema + supplier seed data
│   └── postgres_schema.sql        ← PostgreSQL schema + views
│
├── mlops/
│   └── monitor.py                 ← Prediction logging, drift detection, query log
│
├── knowledge/                     ← Policy documents for the knowledge base tool
└── logs/                          ← Auto-created: predictions.csv, query_log.csv
```

---

---

## Example Queries

The agent handles natural language questions across every domain:

**Inventory & Operations:**
> *"Which products will stock out in the next 7 days?"*
> *"Are there any negative values or unnatural data in our system?"*
> *"Generate a purchase order for all critical items from Royal Canin India"*

**Forecasting:**
> *"Forecast demand for Sara's Chicken & Rice fresh food for the next 30 days"*
> *"Which SKUs have the worst forecast accuracy this month?"*

**Marketing:**
> *"What should our marketing team promote this week based on inventory?"*
> *"What was the inventory impact of the Diwali 2024 sale?"*
> *"Which categories should we markdown to clear before the season ends?"*

**Financial:**
> *"What is the total value of our current inventory in INR?"*
> *"How much revenue are we losing from stockouts?"*

**Product & Customer:**
> *"What food is best for a 4-month-old Labrador Retriever puppy?"*
> *"Which customer segment has the highest lifetime value?"*
> *"What do our Loyal Premium customers buy most?"*

**Supplier:**
> *"Prepare a negotiation brief for Mars Petcare India"*
> *"Which suppliers are getting worse at delivery over the last 6 months?"*

**General Intelligence:**
> *"What are competitors charging for Royal Canin on Amazon.in?"*
> *"Are there any supply chain disruptions affecting pet food imports from Europe?"*

---

## Technology Stack

| Layer | Technology |
|---|---|
| **UI** | Gradio 6, Plotly, Python |
| **Agent** | Custom ReAct loop, Anthropic / OpenAI / Groq / Gemini SDKs |
| **MCP** | FastAPI, uvicorn, SSE, JSON-RPC 2.0 |
| **Primary Forecast** | Temporal Fusion Transformer (pytorch-forecasting 1.7.0, Lightning 2.6.1) |
| **Fallback Forecast** | CatBoost 1.2+ (quantile regression, 3 models) |
| **Databases** | MySQL (aiomysql), PostgreSQL (asyncpg) |
| **Data** | pandas, numpy, 9 synthetic HUFT CSVs |
| **Web Search** | SerpAPI (Google), DuckDuckGo (free fallback) |
| **Python Sandbox** | AST-level security, safe builtins whitelist |
| **MLOps** | CSV-based prediction + drift logging, query audit trail |
| **GPU** | Any CUDA GPU (tested: RTX 4060 8GB) · CUDA 12.x · FP16 mixed precision · CPU fallback |

---

## Design Decisions

**Why MCP instead of RAG?**
Supply chain queries require precise, structured data — exact inventory counts, specific supplier ratings, real demand numbers. RAG retrieves semantically similar text chunks, which introduces imprecision. MCP tools call the actual database and return exact data. There is no approximation.

**Why TFT instead of simpler models?**
HUFT has 24 promotions per year that cause 2–3× demand spikes. A promotion-unaware model will under-forecast during Diwali every time. TFT accepts the promotion calendar as a **known future covariate** — it literally sees "Diwali is in 14 days, 30% discount is planned" during training and learns how demand responds. No other common model handles this natively.

**Why CatBoost as fallback?**
Zero-downtime availability. The TFT takes 30 minutes to train. CatBoost trains in 2 minutes and provides P10/P50/P90 forecasts immediately. The app always has a working forecast engine regardless of TFT training status.

**Why in-process MCP instead of HTTP?**
For a single-machine deployment (local or HF Spaces), in-process MCP eliminates network latency and simplifies deployment. The HTTP transport is available for multi-machine or microservices deployments.

**Why a tool registry instead of a static list + if/elif?**
The old approach required adding a tool in three places: the `MCP_TOOLS` schema list, the `_dispatch_tool_inner` if/elif chain, and the handler map. The new `_TOOL_REGISTRY` dict plus `_build_tool_registry()` is a single source of truth. `dispatch_tool()` routes registered tools with a dict lookup (O(1)) instead of a linear if/elif scan. Adding a new tool now means writing the function and one schema entry — no dispatch code to touch.

**Why TTL caching on CSV tools?**
During a complex multi-step query the agent may call `get_inventory_status` or `get_demand_trends` 3–4 times with the same arguments (to cross-check earlier reasoning). Without caching, each call re-reads and re-computes over 47,515 rows. With 60-second TTL caching, repeated calls within a turn cost ~0ms instead of ~50ms. DB tools (`query_mysql`, `query_postgres`, alerts) are never cached — they always hit the live database to guarantee freshness.

**Why sliding window memory?**
A 20-iteration ReAct loop with verbose tool results can accumulate 50,000+ tokens in the message history — exceeding the context window of most models. The sliding window compresses old tool-result pairs into a compact bullet-point summary (preserving key facts) and keeps the last 6 messages uncompressed. This bounds context growth regardless of query complexity, with no loss of reasoning quality because the compressed summary retains actionable numbers.

---

## Providers Supported

| Provider | Models | Notes |
|---|---|---|
| **Anthropic** | claude-opus-4-5, claude-sonnet-4-5, claude-3-5-haiku | Recommended for complex multi-step reasoning |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | Strong general performance |
| **Groq** | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b | Free tier · Very fast inference |
| **Google Gemini** | gemini-2.0-flash, gemini-1.5-pro, gemini-2.0-flash-lite | Good for long-context queries |

API keys are session-scoped (never stored). Switch providers mid-conversation without restarting.

---

---

## How to Use the App

### Tab 1 — AI Assistant

This is the main interface. Type any supply chain question in plain English and press **Enter**.

**How it works:**
1. Your question is sent to the ReAct agent
2. The agent reasons about which tools to call (shown in grey quoted text as it thinks)
3. It calls live database tools, executes Python, or searches the web as needed
4. It synthesises all results into a final answer

**Tips:**
- You can ask multi-part questions: *"Which SKUs are critically low, and generate a PO for the top 5?"*
- Ask follow-up questions — the agent remembers the full conversation history
- If a tool fails (e.g. database not connected), the agent will automatically try CSV fallback
- Use the **LLM Configuration** panel on the right to switch providers mid-conversation

**Configuring the LLM (right sidebar):**
1. Select a provider: Anthropic / OpenAI / Groq / Gemini
2. Select a model from the dropdown
3. Paste your API key (or leave blank if set in `.env`)
4. Click **Apply** — the agent is now using that provider

**Connecting a database (right sidebar):**
1. Click **MySQL** or **PostgreSQL** tab
2. Toggle **Local** / **Cloud** to pre-fill the fields
3. Add your password
4. Click **Test Connection** — the green checkmark confirms it is live
5. The agent will now query your live database instead of CSV files

---

### Tab 2 — Inventory Dashboard

Real-time inventory risk for all 65 SKUs.

1. Use the **View** dropdown to switch between 5 chart types:
   - **Inventory Health Heatmap** — All SKUs coloured red/amber/green by days of supply
   - **Days of Supply** — Bar chart with CRITICAL/WARNING/OK threshold lines
   - **Inventory vs Demand** — Current stock vs 30-day demand side by side
   - **Stockout Risk Timeline** — Which SKUs stock out and when (next 30 days)
   - **Dead Stock Analysis** — Capital locked in dead/slow/active stock per SKU (₹INR)
2. Use the **Category** dropdown to filter to a product category
3. The **At-Risk SKUs** table below the chart shows all CRITICAL and WARNING items
4. The **Full Inventory Snapshot** table shows all 65 SKUs with risk status
5. The financial KPI cards at the top show total inventory value, dead stock, and lost revenue

---

### Tab 3 — Analytics Dashboard

Three sections, each with switchable charts:

**Marketing Analytics:**
- Select a chart type from the dropdown (Sales by Channel, Brand Performance, etc.)
- Use the **Category** and **Channel** filters to drill down
- The interpretation box below each chart explains what to look for

**Operational Analytics:**
- Lead Time Performance, Cold Chain Monitor, Seasonal Demand Index, Reorder Events
- The Cold Chain Monitor shows temperature breaches and expiry risk for cold-chain SKUs

**Management Dashboard:**
- Financial KPI cards are always visible at the top
- Switch between Private Label vs Third Party, Month-over-Month Growth, and Store Inventory charts
- Each chart has an interpretation panel explaining the business meaning

---

### Tab 4 — Demand Forecast

Per-SKU probabilistic demand forecasting.

1. Select a **SKU** from the dropdown (e.g. `DOG_001 — Royal Canin Adult`)
2. Optionally select a second SKU to compare
3. Set the **forecast horizon** using the slider (7 / 14 / 30 / 90 days) or preset buttons
4. Click **Run Forecast** — or the horizon slider auto-triggers a re-run
5. Read the chart:
   - **Blue line** (left of dashed divider) = last 90 days of actual demand
   - **Green line** (right of divider) = P50 (median) forecast
   - **Red shaded band** = P10–P90 uncertainty range (80% confidence)
   - **Yellow dashed line** = Reorder Point — order when inventory drops below this
   - **Orange shaded blocks** = active promotions in the forecast window
6. The KPI cards below show total forecast in units and estimated ₹INR revenue
7. The **Recommendation** box shows whether to reorder now and suggested quantity

---

### Tab 5 — MLOps Monitor

Model training, forecast accuracy, and drift detection.

**Training a model:**
1. Select a **data source**: CSV (HUFT), MySQL, or PostgreSQL
2. Choose a training mode:
   - **Full TFT Retrain** — trains the Temporal Fusion Transformer from scratch (~30 min on GPU, longer on CPU). Run this once initially and then weekly.
   - **Fine-tune TFT** — updates the existing TFT checkpoint on the last 90 days (~3–8 min). Run after major promotions or new data arrivals.
   - **CatBoost Fallback** — fast tabular model, trains in ~2 min on CPU. Always available even without a GPU.
3. Optionally tick **Fine-tune after full retrain** to do both in sequence
4. Click the training button — progress streams in the log box in real time

**Checking forecast accuracy:**
- The **Forecast Accuracy by SKU** table shows MAPE, MAE, and a grade (A/B/C/D) for every SKU
- Click **Run Drift Check** to compute whether model calibration has degraded
- The drift chart shows MAE over time vs baseline

**Query log:**
- Every AI Assistant query is logged here (provider, model, tools called, duration)
- Use **Clear Log** to reset

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/SCM-using-MCP-and-LLM
cd SCM-using-MCP-and-LLM
pip install -r requirements.txt
```

**For TFT GPU training** (CUDA 12.x, optional but recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-forecasting lightning
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set at least one LLM API key
```

Minimum required — set at least one:
```env
ANTHROPIC_API_KEY=sk-ant-...   # Recommended — best multi-step reasoning
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...           # Free tier available at console.groq.com
GEMINI_API_KEY=AIza...
```

Optional — for live database queries:
```env
# Local MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=pet_store_scm

# Local PostgreSQL
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=your_password
PG_DB=pet_store_scm
```

Optional — for the `web_search` tool:
```env
SERPAPI_KEY=your_key    # Falls back to DuckDuckGo automatically if not set
```

### 3. Generate the HUFT datasets

```bash
python data/generate_data.py
```

Generates all 9 CSV files in `data/` — takes ~30 seconds. The app works fully on CSV alone with no databases required.

### 4. (Optional) Set up databases

**Option A — One command (recommended):**
```bash
python setup_databases.py
```
This auto-detects your credentials from `.env`, runs both MySQL and PostgreSQL setup, seeds all tables from the generated CSVs, and prints a pass/fail summary. Re-running is safe — all inserts use `INSERT IGNORE`.

**Option B — Manual SQL:**
```bash
# MySQL
mysql -u root -p < db/mysql_schema.sql

# PostgreSQL
psql -U postgres -d pet_store_scm -f db/postgres_schema.sql
```

**Option C — Cloud databases (Railway, Supabase, etc.):**

Add your cloud credentials to `.env`:
```env
MYSQL_CLOUD_HOST=interchange.proxy.rlwy.net
MYSQL_CLOUD_PORT=46969
MYSQL_CLOUD_USER=root
MYSQL_CLOUD_PASSWORD=your_railway_password
MYSQL_CLOUD_DB=railway

PG_CLOUD_HOST=nozomi.proxy.rlwy.net
PG_CLOUD_PORT=38110
PG_CLOUD_USER=postgres
PG_CLOUD_PASSWORD=your_railway_password
PG_CLOUD_DB=railway
```

Then in the app's **Database Connection** panel, toggle **Cloud** to pre-fill fields, add your password, and click **Test Connection**.

### 5. Run the app

```bash
python gradio_app.py
```

Open **http://localhost:7860** in your browser.

### 6. Train the forecasting model (optional)

Go to the **MLOps Monitor** tab:
- Click **CatBoost Fallback** first (~2 min, CPU) to get immediate forecasts
- Click **Full TFT Retrain** (~30 min on GPU) for maximum accuracy
- From then on, click **Fine-tune TFT** (~5 min) whenever new data arrives

The app always has a working forecast regardless of training status — CatBoost provides instant coverage.

---

## Troubleshooting

**App starts but charts are blank**
→ Run `python data/generate_data.py` first. The app needs the CSV files.

**"Failed to load MCP tools"**
→ A Python import failed at startup. Check the terminal for the traceback. Most common cause: missing `requirements.txt` package. Run `pip install -r requirements.txt`.

**Agent responds "I don't have access to live database data"**
→ The CSV files are being used (normal if no DB connected). To use live databases, fill in the Database Connection panel in the sidebar and click **Test Connection**.

**MySQL connection refused**
→ Verify MySQL is running: `mysql -u root -p`. Ensure `MYSQL_HOST`, `MYSQL_PORT`, and credentials in `.env` are correct.

**PostgreSQL connection refused**
→ Verify PostgreSQL is running: `psql -U postgres`. Ensure `PG_HOST`, `PG_PORT`, and credentials in `.env` are correct.

**TFT training crashes with CUDA out-of-memory**
→ Reduce `BATCH_SIZE` in `forecasting/ml_forecast.py` line ~480 from 64 to 32, or use CatBoost fallback instead.

**TFT training not available (no pytorch-forecasting)**
→ Install: `pip install pytorch-forecasting lightning`. CatBoost fallback always works without it.

**Groq returns "rate limit exceeded"**
→ Groq free tier has per-minute limits. Switch to Anthropic or wait 60 seconds. Or reduce query complexity.

**Web search returns no results**
→ `SERPAPI_KEY` not set — this is fine, DuckDuckGo fallback is automatic. If DuckDuckGo also fails, it means no internet connectivity from the server.

---

## See Also

- **`TOOLS_README.md`** — Plain-English guide to all 50 tools with example questions
- **`data/generate_data.py`** — Full synthetic data generator with Indian seasonality
- **`forecasting/ml_forecast.py`** — TFT + CatBoost implementation with GPU optimisations
- **`mcp_server/server.py`** — Complete MCP server with all 50 tool implementations
- **`agent/agent.py`** — ReAct agent with full system prompt and graceful degradation logic
