import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"
import Glossary from "../components/Glossary"
import { num } from "../lib/format"

const GLOSSARY = [
  { term: "Sales velocity", what: "How many units of a SKU sell per day, across all stores. The engine of every stockout estimate." },
  { term: "Days to zero", what: "Current inventory ÷ daily velocity — roughly how many days until the SKU runs out if nothing is reordered." },
  { term: "Reorder quantity", what: "How much to reorder now so stock covers the supplier lead time plus a safety buffer before the next delivery lands." },
  { term: "Lead time", what: "Days between placing a purchase order and stock arriving. Longer lead times mean you must reorder earlier." },
  { term: "Risk buckets", what: "Critical = runs out before new stock can arrive · Warning = runs out within lead time + safety buffer · Watch = under 30 days cover · Healthy = comfortable · Excess = over 90 days / not selling." },
]

const RISK_STYLE = {
  critical: "bg-coral/15 text-coral", warning: "bg-[#FF7A45]/15 text-[#FF7A45]",
  watch: "bg-[#FFC53D]/20 text-[#B4820A]", healthy: "bg-teal/15 text-teal",
  excess: "bg-sky/15 text-sky", dead: "bg-ink/10 text-ink/50",
}
const FILTERS = ["all", "critical", "warning", "watch", "healthy", "excess"]

export default function Stockout() {
  const [risk, setRisk] = useState("all")
  const { data, isLoading } = useQuery({
    queryKey: ["stockout", risk],
    queryFn: () => apiGet(`/api/intelligence/stockout${risk === "all" ? "" : `?risk=${risk}`}`),
  })
  const rows = data?.rows ?? []
  const s = data?.summary ?? {}

  const cards = [
    { key: "critical", label: "Critical", emoji: "🔴", accent: "coral" },
    { key: "warning", label: "Warning", emoji: "🟠", accent: "pink" },
    { key: "watch", label: "Watch", emoji: "🟡", accent: "grape" },
    { key: "needs_reorder", label: "Need reorder now", emoji: "📦", accent: "teal" },
  ]

  return (
    <div>
      <PageHeader
        emoji="⏳"
        title="Stockout Predictor"
        blurb="Which products are about to run out, how many days of cover remain, and exactly how much to reorder — ranked by urgency. Velocity-based days-to-zero with lead-time-aware reorder quantities."
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {cards.map((c, i) => (
          <KpiCard
            key={c.key}
            index={i}
            title={c.label}
            value={num(s?.[c.key] ?? 0)}
            accent={c.accent}
            emoji={c.emoji}
            help={c.key === "needs_reorder"
              ? "SKUs in critical/warning that have a positive reorder quantity right now."
              : `SKUs in the ${c.label.toLowerCase()} stockout-risk bucket.`}
          />
        ))}
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setRisk(f)}
            className={`pill capitalize transition ${
              risk === f ? "bg-ink text-white"
                : "bg-white text-ink/65 border border-ink/10 hover:bg-cream"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <ChartCard title="SKUs ranked by stockout urgency"
        hint="Sorted most-urgent first. Reorder quantity covers lead time + a 7-day safety buffer.">
        {isLoading ? (
          <div className="text-ink/40 text-sm py-10 text-center">Loading…</div>
        ) : rows.length === 0 ? (
          <div className="text-ink/40 text-sm py-10 text-center">No SKUs in this bucket.</div>
        ) : (
          <div className="max-h-[560px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="text-left text-ink/50 border-b border-ink/10">
                  <th className="py-2 pr-3 font-bold">SKU</th>
                  <th className="py-2 pr-3 font-bold">Product</th>
                  <th className="py-2 pr-3 font-bold text-right">Inventory</th>
                  <th className="py-2 pr-3 font-bold text-right">Velocity/day</th>
                  <th className="py-2 pr-3 font-bold text-right">Days to zero</th>
                  <th className="py-2 pr-3 font-bold text-right">Reorder qty</th>
                  <th className="py-2 pr-3 font-bold">Risk</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((x) => (
                  <tr key={x.sku_id} className="border-b border-ink/5 hover:bg-cream">
                    <td className="py-2 pr-3 text-ink/50 tabular-nums">{x.sku_id}</td>
                    <td className="py-2 pr-3">
                      <div className="font-semibold text-ink">{x.name}</div>
                      <div className="text-xs text-ink/45">{x.category}</div>
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">{num(x.inventory)}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{num(x.daily_velocity)}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      {x.days_to_zero == null ? "∞" : num(x.days_to_zero)}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums font-semibold">
                      {x.reorder_qty > 0 ? num(x.reorder_qty) : "—"}
                    </td>
                    <td className="py-2 pr-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-bold capitalize ${RISK_STYLE[x.risk] ?? ""}`}>
                        {x.risk}
                      </span>
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
