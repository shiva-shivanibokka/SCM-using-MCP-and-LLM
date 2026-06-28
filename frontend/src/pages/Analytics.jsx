import { useQuery } from "@tanstack/react-query"
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"

const COLORS = ["#0D9488", "#F59E0B", "#F87171", "#1E293B", "#64748B", "#14B8A6"]
const inr = (n) =>
  "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 })

export default function Analytics() {
  const { data } = useQuery({
    queryKey: ["analytics"],
    queryFn: () => apiGet("/api/analytics/overview"),
  })
  const seg = Object.entries(data?.segments ?? {}).map(([name, value]) => ({ name, value }))
  const ret = Object.entries(data?.return_rate_by_category ?? {}).map(
    ([name, value]) => ({ name, value })
  )
  const ltv = Object.entries(data?.ltv_by_segment ?? {})

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Customer Analytics</h1>
      {ltv.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {ltv.map(([seg, v]) => (
            <KpiCard key={seg} title={`LTV · ${seg}`} value={inr(v)} accent="teal" />
          ))}
        </div>
      )}
      <div className="grid md:grid-cols-2 gap-4">
        <ChartCard title="Customer Segments">
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={seg} dataKey="value" nameKey="name" outerRadius={110} label>
                {seg.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Returns by Category">
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={ret} dataKey="value" nameKey="name" outerRadius={110} label>
                {ret.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  )
}
