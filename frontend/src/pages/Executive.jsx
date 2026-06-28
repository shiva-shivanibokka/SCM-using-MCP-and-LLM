import { useQuery } from "@tanstack/react-query"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from "recharts"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"
import { inr, inrCompact } from "../lib/format"

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

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <KpiCard title="Revenue" value={inrCompact(data.revenue)} subtitle="All-time, net" accent="teal" emoji="💰" />
        <KpiCard title="Gross margin" value={`${data.gross_margin_pct}%`} subtitle="After cost of goods" accent="amber" emoji="📈" />
        <KpiCard title="Fill rate" value={`${data.fill_rate}%`} subtitle="Demand met from stock" accent="leaf" emoji="✅" />
        <KpiCard title="Stockout rate" value={`${data.stockout_rate}%`} subtitle="Lower is better" accent="coral" emoji="🚨" />
      </div>

      <div className="mb-6">
        <KpiCard
          title="Inventory value (at cost)"
          value={inr(data.inventory_value)}
          subtitle="Capital tied up in stock — latest snapshot"
          accent="grape"
          emoji="🏷️"
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
    </div>
  )
}
