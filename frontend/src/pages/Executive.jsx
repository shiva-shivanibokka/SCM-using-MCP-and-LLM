import { useQuery } from "@tanstack/react-query"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from "recharts"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"
import Glossary from "../components/Glossary"
import { inr, inrCompact } from "../lib/format"

const GLOSSARY = [
  { term: "Revenue (net)", what: "Total money taken in from sales after discounts, across all stores and channels, for the whole history." },
  { term: "Gross margin %", what: "The share of revenue left after the cost of the goods sold — how profitable sales are before overheads." },
  { term: "Fill rate", what: "The percentage of demand we could satisfy straight from stock. Higher means fewer disappointed customers." },
  { term: "Stockout rate", what: "How often an item had zero demand met because it was out of stock. Lower is better." },
  { term: "Inventory value (at cost)", what: "What the stock currently on shelves is worth at purchase cost — capital tied up in inventory." },
  { term: "Top / Bottom SKUs", what: "The best- and worst-selling individual products by revenue. Protect the top; review the bottom." },
]

const PALETTE = ["#12B5A6", "#FF7A45", "#FF5DA2", "#3DA5F4", "#36C26B", "#8B5CF6", "#FFC53D"]

function SkuList({ rows, danger }) {
  return (
    <ul className="text-sm divide-y divide-ink/5">
      {rows.map((s, i) => (
        <li key={s.sku_id} className="flex items-center justify-between py-1.5">
          <span className="flex items-center gap-2">
            <span className="text-ink/30 font-mono text-xs w-5">{i + 1}</span>
            <span className="font-mono text-xs">{s.sku_id}</span>
          </span>
          <span className={`font-bold ${danger ? "text-coral" : "text-ink"}`}>
            {inrCompact(s.revenue)}
          </span>
        </li>
      ))}
    </ul>
  )
}

export default function Executive() {
  const { data, isLoading } = useQuery({
    queryKey: ["exec-kpis"],
    queryFn: () => apiGet("/api/executive/kpis"),
  })
  if (isLoading || !data)
    return <div className="text-ink/50 animate-pulse">Crunching the numbers…</div>

  const regionData = Object.entries(data.revenue_by_region || {})
    .map(([region, revenue]) => ({ region, revenue }))
    .sort((a, b) => b.revenue - a.revenue)

  return (
    <div>
      <PageHeader
        emoji="📊"
        title="Executive Overview"
        blurb="The one-screen health check: how much we're selling, how profitably, how reliably we keep shelves stocked, and where the money comes from. Everything below is computed live from transactions and inventory."
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-5">
        <KpiCard index={0} title="Revenue" value={inrCompact(data.revenue)} subtitle="All-time, net" accent="teal" emoji="💰"
          help="Total sales after discounts, across every store and channel." />
        <KpiCard index={1} title="Gross margin" value={`${data.gross_margin_pct}%`} subtitle="After cost of goods" accent="amber" emoji="📈"
          help="Profit left after the cost of goods, as a share of revenue." />
        <KpiCard index={2} title="Fill rate" value={`${data.fill_rate}%`} subtitle="Demand met from stock" accent="leaf" emoji="✅"
          help="Share of customer demand served straight from available stock." />
        <KpiCard index={3} title="Stockout rate" value={`${data.stockout_rate}%`} subtitle="Lower is better" accent="coral" emoji="🚨"
          help="How often an item was unavailable when demanded. Lower is better." />
      </div>

      <div className="mb-6">
        <KpiCard
          title="Inventory value (at cost)"
          value={inr(data.inventory_value)}
          subtitle="Capital tied up in stock — latest snapshot"
          accent="grape"
          emoji="🏷️"
          help="The purchase-cost value of all stock currently on shelves."
        />
      </div>

      <ChartCard
        title="Revenue by region"
        hint="Where sales concentrate geographically — useful for distribution and staffing decisions."
        className="mb-6"
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regionData} margin={{ left: 8, right: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2A214010" vertical={false} />
            <XAxis dataKey="region" tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={(v) => inrCompact(v)} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => inr(v)} cursor={{ fill: "#2A214008" }} />
            <Bar dataKey="revenue" radius={[8, 8, 0, 0]}>
              {regionData.map((_, i) => (
                <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="grid md:grid-cols-2 gap-6">
        <ChartCard title="Top SKUs by revenue" hint="Your best sellers — protect availability on these.">
          <SkuList rows={data.top_skus || []} />
        </ChartCard>
        <ChartCard title="Bottom SKUs by revenue" hint="Laggards — candidates for promotion review or delisting.">
          <SkuList rows={data.bottom_skus || []} danger />
        </ChartCard>
      </div>

      <Glossary items={GLOSSARY} />
    </div>
  )
}
