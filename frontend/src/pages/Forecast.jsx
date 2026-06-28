import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"

export default function Forecast() {
  const [sku, setSku] = useState("")
  const [horizon, setHorizon] = useState(30)

  const { data: skus } = useQuery({
    queryKey: ["skus"],
    queryFn: () => apiGet("/api/forecast/skus"),
  })
  const activeSku = sku || skus?.[0]?.sku_id

  const { data: fc, isFetching } = useQuery({
    queryKey: ["fc", activeSku, horizon],
    queryFn: () => apiGet(`/api/forecast/${activeSku}?horizon=${horizon}`),
    enabled: !!activeSku,
  })

  const chart = fc
    ? fc.p50.map((v, i) => ({
        day: i + 1,
        p10: fc.p10[i],
        band: fc.p90[i] - fc.p10[i],
        p50: v,
        p90: fc.p90[i],
      }))
    : []

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Demand Forecast</h1>
      <div className="flex gap-4 items-end flex-wrap">
        <div>
          <label className="text-xs text-navy/60 block">SKU</label>
          <select
            className="rounded-lg border p-2 text-sm min-w-[280px]"
            value={activeSku || ""}
            onChange={(e) => setSku(e.target.value)}
          >
            {skus?.map((s) => (
              <option key={s.sku_id} value={s.sku_id}>
                {s.sku_id} — {s.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-navy/60 block">Horizon</label>
          <select
            className="rounded-lg border p-2 text-sm"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          >
            {[7, 14, 30, 60, 90].map((h) => (
              <option key={h} value={h}>{h} days</option>
            ))}
          </select>
        </div>
      </div>
      <ChartCard
        title={`Ensemble forecast — Chronos + N-HiTS + CatBoost${
          isFetching ? " · updating…" : ""
        }`}
      >
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" label={{ value: "day", position: "insideBottom", offset: -4 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            {/* P10..P90 confidence band via stacked transparent + visible area */}
            <Area dataKey="p10" stackId="band" stroke="none" fill="transparent" name="P10" />
            <Area dataKey="band" stackId="band" stroke="none" fill="#0D9488" fillOpacity={0.15} name="P10–P90" />
            <Line type="monotone" dataKey="p50" stroke="#0D9488" strokeWidth={2} dot={false} name="P50 (median)" />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}
