import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import Glossary from "../components/Glossary"
import { num, titleize } from "../lib/format"

const GLOSSARY = [
  { term: "On-time delivery", what: "Share of orders that arrived by the promised date. Below ~90% disrupts replenishment planning." },
  { term: "Fill rate", what: "Share of each order the supplier could actually ship in full — partial shipments hurt our shelf availability." },
  { term: "Quality rating", what: "A 0–5 score for goods condition and spec compliance on arrival. Higher is better." },
  { term: "Defect rate", what: "Percentage of received units that were damaged or out-of-spec. Lower is better." },
  { term: "Lead time", what: "Average days from placing an order to receiving it. Shorter lets us hold less safety stock." },
  { term: "Colour grading", what: "Green = on target, amber = watch, red = needs a conversation. Thresholds differ per metric (e.g. low defects are good, high on-time is good)." },
]

// metric → how to read it. higherIsBetter drives the color grading.
const METRICS = [
  { key: "on_time_delivery_pct", label: "On-time delivery", unit: "%", emoji: "⏱️", higher: true, good: 95, ok: 90 },
  { key: "fill_rate_pct", label: "Fill rate", unit: "%", emoji: "📦", higher: true, good: 97, ok: 92 },
  { key: "quality_rating", label: "Quality rating", unit: "/5", emoji: "⭐", higher: true, good: 4.3, ok: 3.8 },
  { key: "defect_rate_pct", label: "Defect rate", unit: "%", emoji: "🔧", higher: false, good: 2, ok: 4 },
  { key: "lead_time_actual_days", label: "Lead time", unit: "d", emoji: "🚚", higher: false, good: 5, ok: 9 },
]

function grade(m, v) {
  if (v == null) return "ink"
  if (m.higher) return v >= m.good ? "leaf" : v >= m.ok ? "amber" : "coral"
  return v <= m.good ? "leaf" : v <= m.ok ? "amber" : "coral"
}
const TINT = { leaf: "bg-leaf/10 text-leaf", amber: "bg-amber/10 text-amber", coral: "bg-coral/10 text-coral", ink: "bg-ink/10 text-ink" }

export default function Suppliers() {
  const { data } = useQuery({
    queryKey: ["suppliers"],
    queryFn: () => apiGet("/api/suppliers/scorecard"),
  })
  const rows = data?.suppliers ?? []
  const [picked, setPicked] = useState("")
  const active = rows.find((r) => r.supplier_name === picked) || rows[0]

  // rank by on-time delivery for context
  const ranked = [...rows].sort(
    (a, b) => (b.on_time_delivery_pct ?? 0) - (a.on_time_delivery_pct ?? 0)
  )
  const rank = active ? ranked.findIndex((r) => r.supplier_name === active.supplier_name) + 1 : 0

  return (
    <div>
      <PageHeader
        emoji="🚚"
        title="Supplier Scorecard"
        blurb="Pick a supplier to see how they perform on the things that matter for replenishment — delivery reliability, fill rate, quality, defects, and lead time. Green is on target, amber is watch, red needs a conversation."
      >
        {rows.length > 0 && (
          <select
            className="rounded-xl border border-ink/15 bg-white px-4 py-2.5 text-sm font-bold shadow-pop"
            value={active?.supplier_name || ""}
            onChange={(e) => setPicked(e.target.value)}
          >
            {ranked.map((r) => (
              <option key={r.supplier_name} value={r.supplier_name}>
                {r.supplier_name}
              </option>
            ))}
          </select>
        )}
      </PageHeader>

      {!active ? (
        <div className="text-ink/50 animate-pulse">Loading suppliers…</div>
      ) : (
        <>
          <motion.div
            key={active.supplier_name}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-6 mb-6 flex items-center gap-4"
          >
            <div className="h-14 w-14 grid place-items-center rounded-2xl bg-teal/10 text-3xl">
              🏭
            </div>
            <div>
              <div className="font-display text-2xl font-700 text-ink">
                {active.supplier_name}
              </div>
              <div className="text-sm text-ink/55">
                Ranked <b className="text-teal">#{rank}</b> of {rows.length} by on-time delivery
              </div>
            </div>
          </motion.div>

          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            {METRICS.filter((m) => active[m.key] != null).map((m, i) => {
              const v = active[m.key]
              const g = grade(m, v)
              return (
                <motion.div
                  key={m.key}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className="card p-5 text-center"
                >
                  <div className={`mx-auto h-12 w-12 grid place-items-center rounded-full text-xl ${TINT[g]}`}>
                    {m.emoji}
                  </div>
                  <div className="font-display text-2xl font-700 text-ink mt-3">
                    {num(v)}
                    <span className="text-sm text-ink/40 font-sans">{m.unit}</span>
                  </div>
                  <div className="text-xs font-bold text-ink/55 mt-1">{m.label}</div>
                </motion.div>
              )
            })}
          </div>
        </>
      )}

      <Glossary items={GLOSSARY} />
    </div>
  )
}
