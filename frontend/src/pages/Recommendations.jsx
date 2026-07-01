import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from "recharts"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"
import Glossary from "../components/Glossary"
import { num } from "../lib/format"

const GLOSSARY = [
  { term: "Frequently bought together", what: "Product pairs that show up in the same customer order more often than the rest. The basis of cross-sell and 'customers who bought X also bought Y' suggestions." },
  { term: "Co-purchases", what: "How many separate orders contained both products. Higher = a stronger, more reliable pairing." },
  { term: "Support %", what: "The share of all orders that contained both products together. A size-independent way to compare how common a pairing is." },
  { term: "Recommend for a product", what: "Pick any product and see what customers most often buy alongside it — the exact list you'd surface on a product page or in a 'complete the set' email." },
  { term: "How it's built", what: "A dbt + DuckDB mart pre-computes every pairing from the full order history, so a recommendation is an instant lookup no matter how large the data grows." },
]

const PALETTE = ["#12B5A6", "#FF7A45", "#FF5DA2", "#3DA5F4", "#36C26B", "#8B5CF6", "#FFC53D"]
const selectCls = "px-3 py-2 rounded-xl border border-ink/15 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-teal/40"
const short = (s) => (String(s ?? "").length > 22 ? String(s).slice(0, 21) + "…" : String(s ?? ""))

export default function Recommendations() {
  const { data, isLoading } = useQuery({
    queryKey: ["recommendations"],
    queryFn: () => apiGet("/api/recommendations/overview"),
  })
  const { data: opts } = useQuery({
    queryKey: ["intel-options"],
    queryFn: () => apiGet("/api/intelligence/options"),
  })

  const [category, setCategory] = useState("")
  const [sku, setSku] = useState("")
  const [picked, setPicked] = useState(null)   // set on button click

  const forSku = useQuery({
    queryKey: ["rec-for-sku", picked],
    queryFn: () => apiGet(`/api/recommendations/for-sku/${encodeURIComponent(picked)}`),
    enabled: !!picked,
  })

  const skus = (opts?.skus ?? []).filter((s) => !category || s.category === category)
  const pairs = data?.pairs ?? []
  const chart = pairs.slice(0, 12).map((p) => ({
    name: `${short(p.sku_a_name)} + ${short(p.sku_b_name)}`,
    value: p.co_purchases,
  }))

  return (
    <div>
      <PageHeader
        emoji="🎯"
        title="Product Recommendations"
        blurb="Which products customers buy together, learned from real order baskets. Pick a product for its cross-sell list, or browse the strongest pairings overall. Pre-computed by a dbt + DuckDB mart, so it stays instant as the data grows."
      />

      {/* Interactive: recommend for a chosen product */}
      <ChartCard title="Recommend for a product"
        hint="Choose a category to narrow the list, pick a product, then get its 'bought together' recommendations.">
        <div className="flex flex-wrap items-end gap-4 py-1">
          <label className="flex flex-col gap-1">
            <span className="text-sm font-bold text-ink/60">Category</span>
            <select value={category}
              onChange={(e) => { setCategory(e.target.value); setSku("") }}
              className={selectCls + " w-52"}>
              <option value="">All categories</option>
              {(opts?.categories ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-sm font-bold text-ink/60">Product</span>
            <select value={sku} onChange={(e) => setSku(e.target.value)} className={selectCls + " w-72"}>
              <option value="">Select a product…</option>
              {skus.map((s) => <option key={s.sku_id} value={s.sku_id}>{s.name} ({s.sku_id})</option>)}
            </select>
          </label>
          <button onClick={() => sku && setPicked(sku)} disabled={!sku}
            className="pill bg-ink text-white hover:bg-ink/90 disabled:opacity-40">
            🎯 Get recommendations
          </button>
        </div>

        {picked && (
          <div className="mt-5">
            {forSku.isFetching ? (
              <div className="text-ink/40 text-sm py-6 text-center">Finding recommendations…</div>
            ) : (forSku.data?.recommendations ?? []).length === 0 ? (
              <div className="text-ink/40 text-sm py-6 text-center">
                No co-purchase data for this product yet.
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-ink/50 border-b border-ink/10">
                    <th className="py-2 pr-3 font-bold">Customers who bought this also bought</th>
                    <th className="py-2 pr-3 font-bold">Category</th>
                    <th className="py-2 pr-3 font-bold text-right">Co-purchases</th>
                    <th className="py-2 pr-3 font-bold text-right">Support %</th>
                  </tr>
                </thead>
                <tbody>
                  {forSku.data.recommendations.map((x, i) => (
                    <tr key={i} className="border-b border-ink/5 hover:bg-cream">
                      <td className="py-2 pr-3 font-semibold text-ink">{x.name}</td>
                      <td className="py-2 pr-3 text-ink/55">{x.category}</td>
                      <td className="py-2 pr-3 text-right tabular-nums">{num(x.co_purchases)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums font-semibold text-teal">
                        {Number(x.support_pct ?? 0).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </ChartCard>

      {/* Overview KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 my-6">
        <KpiCard index={0} title="Product pairings found" value={num(data?.total_pairs ?? 0)}
          accent="teal" emoji="🔗" help="Distinct product pairs bought together often enough to recommend." />
        <KpiCard index={1} title="Strongest pairing" value={num(data?.max_co_purchases ?? 0)}
          accent="pink" emoji="⭐" help="Co-purchase count of the single most frequently bought-together pair." />
        <KpiCard index={2} title="Top pair" value={data?.top_pair ?? "—"}
          accent="sky" emoji="🛒" help="The two products most often bought in the same order." />
        <KpiCard index={3} title="Source"
          value={data?.source === "dbt_mart" ? "dbt mart" : "live compute"}
          accent="grape" emoji="⚙️" help="Whether these came from the pre-built dbt + DuckDB mart or were computed on the fly." />
      </div>

      <ChartCard title="Top frequently-bought-together pairs"
        hint="Ranked by how many orders contained both products. Taller bars are stronger cross-sell opportunities.">
        {isLoading ? (
          <div className="text-ink/40 text-sm py-10 text-center">Loading…</div>
        ) : chart.length === 0 ? (
          <div className="text-ink/40 text-sm py-10 text-center">No pairings available.</div>
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(320, chart.length * 40)}>
            <BarChart data={chart} layout="vertical" margin={{ left: 24, right: 32 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A214010" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="name" width={220} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [`${v.toLocaleString("en-IN")} orders`, "Co-purchases"]}
                cursor={{ fill: "#2A214008" }} />
              <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                {chart.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      <ChartCard title="Frequently bought together"
        hint="Cross-sell table — put high-support pairs near each other, or bundle them in promotions.">
        {pairs.length === 0 ? (
          <div className="text-ink/40 text-sm py-10 text-center">No pairings available.</div>
        ) : (
          <div className="max-h-[520px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="text-left text-ink/50 border-b border-ink/10">
                  <th className="py-2 pr-3 font-bold">#</th>
                  <th className="py-2 pr-3 font-bold">Product A</th>
                  <th className="py-2 pr-3 font-bold">Product B</th>
                  <th className="py-2 pr-3 font-bold text-right">Co-purchases</th>
                  <th className="py-2 pr-3 font-bold text-right">Support %</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((p, i) => (
                  <tr key={i} className="border-b border-ink/5 hover:bg-cream">
                    <td className="py-2 pr-3 text-ink/40">{i + 1}</td>
                    <td className="py-2 pr-3">
                      <div className="font-semibold text-ink">{p.sku_a_name}</div>
                      <div className="text-xs text-ink/45">{p.sku_a_category}</div>
                    </td>
                    <td className="py-2 pr-3">
                      <div className="font-semibold text-ink">{p.sku_b_name}</div>
                      <div className="text-xs text-ink/45">{p.sku_b_category}</div>
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">{num(p.co_purchases)}</td>
                    <td className="py-2 pr-3 text-right tabular-nums font-semibold text-teal">
                      {Number(p.support_pct ?? 0).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </ChartCard>

      <Glossary items={GLOSSARY} />
    </div>
  )
}
