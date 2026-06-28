import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import { num } from "../lib/format"

const CATS = [
  { key: "stockout_risk", label: "Stockout Risk", emoji: "🚨", accent: "coral",
    hint: "Less than 7 days of cover — reorder now or risk going out of stock." },
  { key: "reorder_list", label: "Reorder Soon", emoji: "🛒", accent: "amber",
    hint: "Under 14 days of cover — schedule a purchase order this week." },
  { key: "dead_stock", label: "Dead Stock", emoji: "🪦", accent: "grape",
    hint: "Inventory on hand but no recent demand — candidates for markdown or clearance." },
  { key: "overstock", label: "Overstock", emoji: "📚", accent: "sky",
    hint: "More than 90 days of cover — capital tied up; slow down replenishment." },
]

const BAR = { coral: "bg-coral", amber: "bg-amber", grape: "bg-grape", sky: "bg-sky" }
const TEXT = { coral: "text-coral", amber: "text-amber", grape: "text-grape", sky: "text-sky" }

export default function Inventory() {
  const { data, isLoading } = useQuery({
    queryKey: ["inv-health"],
    queryFn: () => apiGet("/api/inventory/health"),
  })
  const [active, setActive] = useState("stockout_risk")

  if (isLoading || !data)
    return <div className="text-ink/50 animate-pulse">Fetching inventory…</div>

  const cat = CATS.find((c) => c.key === active)
  const rows = data[active] ?? []

  return (
    <div>
      <PageHeader
        emoji="📦"
        title="Inventory Health"
        blurb="Every SKU's latest cover position, bucketed by urgency. Tap a card to load that watchlist into the table below — one place to triage what to reorder, clear, or slow down."
      />

      {/* Clickable category tiles */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {CATS.map((c) => {
          const count = (data[c.key] ?? []).length
          const isActive = c.key === active
          return (
            <motion.button
              key={c.key}
              onClick={() => setActive(c.key)}
              whileHover={{ y: -4 }}
              whileTap={{ scale: 0.97 }}
              className={`card text-left overflow-hidden ring-2 transition ${
                isActive ? "ring-ink/80" : "ring-transparent"
              }`}
            >
              <div className={`h-1.5 ${BAR[c.accent]}`} />
              <div className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-extrabold uppercase tracking-wide text-ink/55">
                    {c.label}
                  </span>
                  <span className="text-xl">{c.emoji}</span>
                </div>
                <div className="font-display text-3xl font-700 text-ink mt-1">
                  {count}
                </div>
                <div className="text-[11px] text-ink/45 mt-1">SKUs · tap to view</div>
              </div>
            </motion.button>
          )
        })}
      </div>

      {/* Single table, swaps content on tab click */}
      <div className="card overflow-hidden">
        <div className="flex items-center gap-2 px-5 pt-4">
          <span className="text-xl">{cat.emoji}</span>
          <div>
            <div className={`font-display text-lg font-600 ${TEXT[cat.accent]}`}>
              {cat.label}
              <span className="text-ink/40 font-sans text-sm font-normal"> · {rows.length} SKUs</span>
            </div>
            <div className="text-xs text-ink/50">{cat.hint}</div>
          </div>
        </div>

        <div className="mt-3 max-h-[460px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-white/95 backdrop-blur z-10">
              <tr className="text-left text-ink/50 border-b border-ink/10">
                <th className="py-2 px-5 font-bold">SKU</th>
                <th className="py-2 px-3 font-bold">Name</th>
                <th className="py-2 px-3 font-bold text-right">On hand</th>
                <th className="py-2 px-5 font-bold text-right">Days of cover</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={4} className="py-10 text-center text-ink/40">
                    Nothing in this bucket right now 🎉
                  </td>
                </tr>
              )}
              {rows.map((r) => (
                <tr key={r.sku_id} className="border-b border-ink/5 hover:bg-cream/60">
                  <td className="py-2 px-5 font-mono text-xs">{r.sku_id}</td>
                  <td className="py-2 px-3 text-ink/70">{r.name || "—"}</td>
                  <td className="py-2 px-3 text-right">{num(r.inventory)}</td>
                  <td className={`py-2 px-5 text-right font-bold ${TEXT[cat.accent]}`}>
                    {r.days_of_cover >= 999 ? "∞" : num(r.days_of_cover)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
