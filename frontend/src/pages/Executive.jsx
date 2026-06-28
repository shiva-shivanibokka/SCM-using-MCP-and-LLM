import { useQuery } from "@tanstack/react-query"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { apiGet } from "../lib/api"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"

const inr = (n) =>
  "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 })

export default function Executive() {
  const { data, isLoading } = useQuery({
    queryKey: ["exec-kpis"],
    queryFn: () => apiGet("/api/executive/kpis"),
  })
  if (isLoading || !data) return <div className="text-navy/50">Loading…</div>

  const regionData = Object.entries(data.revenue_by_region).map(
    ([region, revenue]) => ({ region, revenue })
  )

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Executive Overview</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard title="Revenue" value={inr(data.revenue)} subtitle="All-time" accent="teal" />
        <KpiCard title="Gross Margin" value={`${data.gross_margin_pct}%`} accent="amber" />
        <KpiCard title="Fill Rate" value={`${data.fill_rate}%`} accent="teal" />
        <KpiCard title="Stockout Rate" value={`${data.stockout_rate}%`} accent="coral" />
      </div>
      <KpiCard
        title="Inventory Value (cost)"
        value={inr(data.inventory_value)}
        subtitle="Latest snapshot"
        accent="amber"
      />
      <ChartCard title="Revenue by Region">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regionData}>
            <XAxis dataKey="region" />
            <YAxis />
            <Tooltip formatter={(v) => inr(v)} />
            <Bar dataKey="revenue" fill="#0D9488" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <div className="grid md:grid-cols-2 gap-4">
        <ChartCard title="Top 10 SKUs by Revenue">
          <ul className="text-sm space-y-1">
            {data.top_skus.map((s) => (
              <li key={s.sku_id} className="flex justify-between border-b py-1">
                <span>{s.sku_id}</span>
                <span className="font-semibold">{inr(s.revenue)}</span>
              </li>
            ))}
          </ul>
        </ChartCard>
        <ChartCard title="Bottom 10 SKUs by Revenue">
          <ul className="text-sm space-y-1">
            {data.bottom_skus.map((s) => (
              <li key={s.sku_id} className="flex justify-between border-b py-1">
                <span>{s.sku_id}</span>
                <span className="font-semibold text-coral">{inr(s.revenue)}</span>
              </li>
            ))}
          </ul>
        </ChartCard>
      </div>
    </div>
  )
}
