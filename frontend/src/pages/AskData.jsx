import { useState } from "react"
import { apiPost } from "../lib/api"
import { useLLMStore } from "../stores/llmStore"
import PageHeader from "../components/PageHeader"
import ChartCard from "../components/ChartCard"
import LlmSelector from "../components/LlmSelector"
import Glossary from "../components/Glossary"

const EXAMPLES = [
  { label: "Top 5 cities by revenue", sql: "SELECT city, ROUND(SUM(net_revenue_inr)) AS revenue\nFROM transactions GROUP BY city ORDER BY revenue DESC LIMIT 5" },
  { label: "Revenue by category", sql: "SELECT p.category, ROUND(SUM(t.net_revenue_inr)) AS revenue\nFROM transactions t JOIN products p ON t.sku_id = p.sku_id\nGROUP BY p.category ORDER BY revenue DESC" },
  { label: "Biggest baskets", sql: "SELECT order_id, COUNT(*) AS items, ROUND(SUM(net_revenue_inr)) AS value\nFROM transactions GROUP BY order_id ORDER BY items DESC, value DESC LIMIT 10" },
  { label: "Repeat customers", sql: "SELECT customer_id, COUNT(DISTINCT order_id) AS orders\nFROM transactions GROUP BY customer_id ORDER BY orders DESC LIMIT 10" },
]

const SCHEMA = [
  ["transactions", "order_id, customer_id, date, sku_id, brand, category, quantity, unit_price_inr, discount_pct, net_revenue_inr, channel, city, customer_segment, store_id"],
  ["products", "sku_id, name, brand, category, pet_type, life_stage, price_inr, cost_inr, margin_pct, supplier, lead_time_days"],
  ["stores", "store_id, city, state, region, store_type"],
  ["demand", "sku_id, date, demand, inventory"],
  ["customers", "customer_id, city, segment, pet_type, lifetime_value_inr"],
  ["store_inventory", "date, store_id, sku_id, name, category, demand, inventory, lead_time_days, days_of_supply, risk_status"],
  ["suppliers", "supplier_name, on_time_delivery_pct, defect_rate_pct, fill_rate_pct"],
  ["returns", "return_id, sku_id, category, return_reason, refund_inr"],
]

const GLOSSARY = [
  { term: "Ask Your Data", what: "Run your own read-only SQL against the warehouse for questions the fixed dashboards don't answer. The AI Assistant can also write this SQL for you from a plain-English question." },
  { term: "Read-only & guarded", what: "Only a single SELECT is allowed — inserts, updates, deletes, and file access are blocked, and results are capped at 100 rows. You can explore freely without any risk to the data." },
  { term: "GMV vs NR", what: "GMV = unit_price_inr × quantity (full sticker value); NR = net_revenue_inr (after discount)." },
]

export default function AskData() {
  const [sql, setSql] = useState(EXAMPLES[0].sql)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [question, setQuestion] = useState("")
  const [asking, setAsking] = useState(false)
  const { provider, model, apiKey } = useLLMStore()

  async function run() {
    setLoading(true)
    try {
      setResult(await apiPost("/api/intelligence/sql", { sql }))
    } catch (e) {
      setResult({ error: String(e), columns: [], rows: [] })
    } finally {
      setLoading(false)
    }
  }

  // Plain-English → SQL: the LLM writes the query into the editor, and we run it.
  async function ask() {
    if (!question.trim()) return
    setAsking(true)
    try {
      const res = await apiPost("/api/intelligence/ask", {
        question, provider, model, api_key: apiKey,
      })
      if (res.sql) setSql(res.sql)   // drop the generated SQL into the editor (editable)
      setResult(res)
    } catch (e) {
      setResult({ error: String(e), columns: [], rows: [] })
    } finally {
      setAsking(false)
    }
  }

  return (
    <div>
      <PageHeader
        emoji="🔎"
        title="Ask Your Data"
        blurb="Run ad-hoc SQL against the warehouse for anything the dashboards don't cover. Read-only and guarded — explore freely. Prefer plain English? The AI Assistant writes this SQL for you."
      />

      <div className="mb-6">
        <LlmSelector />
        <ChartCard title="Ask in plain English"
          hint="Type a question — the model writes the SQL into the editor below, then runs it. You can tweak the SQL and re-run.">
          <div className="flex flex-wrap items-center gap-3 py-1">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
              placeholder="e.g. top 5 categories by net revenue this year"
              className="flex-1 min-w-[240px] px-3 py-2 rounded-xl border border-ink/15 text-sm focus:outline-none focus:ring-2 focus:ring-teal/40"
            />
            <button onClick={ask} disabled={asking || !question.trim()}
              className="pill bg-teal text-white hover:bg-teal/90 disabled:opacity-40">
              {asking ? "Thinking…" : "✨ Generate & run"}
            </button>
          </div>
          <p className="text-[11px] text-ink/45 mt-2 px-1">
            Uses your selected provider/key above. The generated SQL appears in the editor so you can verify or edit it.
          </p>
        </ChartCard>
      </div>

      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        <div>
          <ChartCard title="SQL query" hint="A single read-only SELECT. Try an example, then edit it.">
            <div className="flex flex-wrap gap-2 mb-3">
              {EXAMPLES.map((ex) => (
                <button key={ex.label} onClick={() => setSql(ex.sql)}
                  className="pill bg-white text-ink/65 border border-ink/10 hover:bg-cream text-xs">
                  {ex.label}
                </button>
              ))}
            </div>
            <textarea
              value={sql}
              onChange={(e) => setSql(e.target.value)}
              spellCheck={false}
              rows={6}
              className="w-full font-mono text-sm px-3 py-2 rounded-xl border border-ink/15 bg-cream/40 focus:outline-none focus:ring-2 focus:ring-teal/40"
            />
            <div className="mt-3">
              <button onClick={run} disabled={loading}
                className="pill bg-ink text-white hover:bg-ink/90 disabled:opacity-50">
                {loading ? "Running…" : "▶ Run query"}
              </button>
            </div>
          </ChartCard>

          {result && (
            <div className="mt-6">
              {result.error ? (
                <div className="px-4 py-3 rounded-xl bg-coral/12 text-coral text-sm font-medium">
                  {result.error}
                </div>
              ) : (
                <ChartCard
                  title="Results"
                  hint={`${result.total} row${result.total === 1 ? "" : "s"}${result.truncated ? " (showing first 100)" : ""}`}
                >
                  {result.rows.length === 0 ? (
                    <div className="text-ink/40 text-sm py-8 text-center">No rows returned.</div>
                  ) : (
                    <div className="max-h-[460px] overflow-auto">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-white">
                          <tr className="text-left text-ink/50 border-b border-ink/10">
                            {result.columns.map((c) => (
                              <th key={c} className="py-2 pr-3 font-bold whitespace-nowrap">{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {result.rows.map((row, i) => (
                            <tr key={i} className="border-b border-ink/5 hover:bg-cream">
                              {result.columns.map((c) => (
                                <td key={c} className="py-2 pr-3 tabular-nums whitespace-nowrap">
                                  {row[c] == null ? "—" : String(row[c])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </ChartCard>
              )}
            </div>
          )}
        </div>

        <ChartCard title="Tables" hint="Click a table to start a query.">
          <div className="space-y-3 text-xs">
            {SCHEMA.map(([table, cols]) => (
              <div key={table}>
                <button
                  onClick={() => setSql(`SELECT * FROM ${table} LIMIT 20`)}
                  className="font-bold text-teal hover:underline"
                >
                  {table}
                </button>
                <div className="text-ink/45 leading-relaxed">{cols}</div>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>

      <Glossary items={GLOSSARY} />
    </div>
  )
}
