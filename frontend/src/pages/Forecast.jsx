import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend,
} from "recharts"
import { Play, Loader2 } from "lucide-react"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"
import { num } from "../lib/format"

const COMPCOLORS = { chronos: "#12B5A6", nhits: "#8B5CF6", catboost: "#FF7A45" }

export default function Forecast() {
  const [sku, setSku] = useState("")
  const [horizon, setHorizon] = useState(30)
  const [showComponents, setShowComponents] = useState(false)
  const [ran, setRan] = useState(false) // only fetch after Run is clicked

  const { data: skus } = useQuery({
    queryKey: ["skus"],
    queryFn: () => apiGet("/api/forecast/skus"),
  })
  const activeSku = sku || skus?.[0]?.sku_id

  const { data: fc, isFetching, refetch } = useQuery({
    queryKey: ["fc", activeSku, horizon],
    queryFn: () => apiGet(`/api/forecast/${activeSku}?horizon=${horizon}&components=true`),
    enabled: false,
  })

  const run = () => {
    setRan(true)
    refetch()
  }

  const chart = fc
    ? fc.p50.map((v, i) => ({
        day: i + 1,
        p10: fc.p10[i],
        band: fc.p90[i] - fc.p10[i],
        p50: v,
        chronos: fc.components?.chronos?.p50?.[i],
        nhits: fc.components?.nhits?.p50?.[i],
        catboost: fc.components?.catboost?.p50?.[i],
      }))
    : []

  const total = fc ? fc.p50.reduce((a, b) => a + b, 0) : 0
  const peak = fc ? Math.max(...fc.p50) : 0
  const peakDay = fc ? fc.p50.indexOf(peak) + 1 : 0
  const avg = fc ? total / fc.p50.length : 0
  const activeModels = fc?.components ? Object.keys(fc.components) : []

  return (
    <div>
      <PageHeader
        emoji="🔮"
        title="Demand Forecast"
        blurb="A zero-shot ensemble — Chronos-T5 (foundation model) + N-HiTS + CatBoost — predicts daily demand with P10–P90 confidence. Pick a SKU and horizon, then run it. The shaded band is the uncertainty range; the line is the expected demand."
      />

      {/* Controls + Run button */}
      <div className="card p-4 mb-6">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="text-xs font-bold text-ink/60 block mb-1">SKU</label>
            <select
              className="rounded-xl border border-ink/15 bg-cream px-3 py-2 text-sm min-w-[280px]"
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
            <label className="text-xs font-bold text-ink/60 block mb-1">Horizon</label>
            <select
              className="rounded-xl border border-ink/15 bg-cream px-3 py-2 text-sm"
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
            >
              {[7, 14, 30, 60, 90].map((h) => (
                <option key={h} value={h}>{h} days</option>
              ))}
            </select>
          </div>
          <button
            onClick={run}
            disabled={isFetching || !activeSku}
            className="bg-teal text-white font-bold px-5 py-2.5 rounded-xl flex items-center gap-2 disabled:opacity-50 hover:brightness-95 transition shadow-pop"
          >
            {isFetching ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            {isFetching ? "Forecasting…" : "Run Forecast"}
          </button>
          {fc && (
            <label className="flex items-center gap-2 text-sm text-ink/70 ml-auto cursor-pointer">
              <input
                type="checkbox"
                checked={showComponents}
                onChange={(e) => setShowComponents(e.target.checked)}
              />
              Show model breakdown
            </label>
          )}
        </div>
        {fc && (
          <p className="text-[11px] text-ink/45 mt-3">
            Active models this run: {activeModels.join(" · ") || "—"}. The Chronos
            foundation model runs only on the deployed backend; the API degrades
            gracefully to whatever components are available.
          </p>
        )}
      </div>

      {!ran && (
        <div className="card p-12 text-center">
          <div className="text-5xl animate-bob">🔮</div>
          <p className="text-ink/50 mt-3">
            Choose a SKU and hit <b>Run Forecast</b> to see the demand outlook.
          </p>
        </div>
      )}

      {ran && fc && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <KpiCard title={`Total demand (${horizon}d)`} value={num(total)} accent="teal" emoji="📈" />
            <KpiCard title="Avg / day" value={num(avg)} accent="sky" emoji="📅" />
            <KpiCard title="Peak day" value={`Day ${peakDay}`} subtitle={`${num(peak)} units`} accent="orange" emoji="⛰️" />
            <KpiCard title="Models blended" value={activeModels.length} accent="grape" emoji="🧩" />
          </div>

          <ChartCard
            title="Demand outlook with P10–P90 confidence"
            hint="Shaded teal = the band where demand will likely land. The solid line is the median (P50) expectation."
          >
            <ResponsiveContainer width="100%" height={380}>
              <ComposedChart data={chart} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A214012" />
                <XAxis dataKey="day" tick={{ fontSize: 12 }}
                  label={{ value: "day", position: "insideBottom", offset: -2, fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Area dataKey="p10" stackId="band" stroke="none" fill="transparent" name="P10" />
                <Area dataKey="band" stackId="band" stroke="none" fill="#12B5A6" fillOpacity={0.16} name="P10–P90" />
                <Line type="monotone" dataKey="p50" stroke="#12B5A6" strokeWidth={2.5} dot={false} name="P50 (median)" />
                {showComponents &&
                  activeModels.map((m) => (
                    <Line key={m} type="monotone" dataKey={m} stroke={COMPCOLORS[m]}
                      strokeWidth={1.5} strokeDasharray="4 3" dot={false} name={m} />
                  ))}
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>
        </>
      )}
    </div>
  )
}
