import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { Store } from "lucide-react"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import Glossary from "../components/Glossary"
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

const GLOSSARY = [
  { term: "Days of cover", what: "How many days the current stock will last at recent demand. 10 = about ten days before it runs out." },
  { term: "Stockout risk", what: "Items with under 7 days of cover — the most urgent to reorder before shelves go empty." },
  { term: "Reorder soon", what: "Items under 14 days of cover — fine for now, but should be on this week's purchase order." },
  { term: "Dead stock", what: "Items sitting in inventory with no recent demand — tying up cash and shelf space; consider clearance." },
  { term: "Overstock", what: "Items with more than 90 days of cover — over-ordered; ease off replenishment to free up capital." },
  { term: "Store scope", what: "‘All stores (national)’ aggregates demand company-wide. Pick a store to see that location's own watchlists." },
]

export default function Inventory() {
  const [storeId, setStoreId] = useState("") // "" = all stores
  const [active, setActive] = useState("stockout_risk")

  const { data: storeOpts } = useQuery({
    queryKey: ["inv-stores"],
    queryFn: () => apiGet("/api/inventory/stores"),
  })
  const { data, isLoading } = useQuery({
    queryKey: ["inv-health", storeId],
    queryFn: () => apiGet(`/api/inventory/health${storeId ? `?store_id=${storeId}` : ""}`),
  })

  const cat = CATS.find((c) => c.key === active)
  const rows = data?.[active] ?? []
  const scopeLabel = storeId
    ? storeOpts?.stores.find((s) => s.store_id === storeId)?.label || storeId
    : "All stores (national)"

  return (
    <div>
      <PageHeader
        emoji="📦"
        title="Inventory Health"
        blurb="Every SKU's latest cover position, bucketed by urgency. Choose a store to triage that location, or stay national. Tap a card to load that watchlist into the table below."
      >
        <div className="flex items-center gap-2 rounded-2xl border border-ink/15 bg-white px-3 py-2 shadow-pop">
          <Store size={18} className="text-teal" />
          <select
            className="bg-transparent text-base font-bold outline-none"
            value={storeId}
            onChange={(e) => setStoreId(e.target.value)}
          >
            <option value="">All stores (national)</option>
            {storeOpts?.stores.map((s) => (
              <option key={s.store_id} value={s.store_id}>{s.label}</option>
            ))}
          </select>
        </div>
      </PageHeader>

      <div className="mb-4 inline-flex items-center gap-2 pill bg-teal/10 text-teal">
        <Store size={15} /> Showing: {scopeLabel}
      </div>

      {isLoading || !data ? (
        <div className="text-ink/50 animate-pulse text-lg">Fetching inventory…</div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-6">
            {CATS.map((c, i) => {
              const count = (data[c.key] ?? []).length
              const isActive = c.key === active
              return (
                <motion.button
                  key={c.key}
                  onClick={() => setActive(c.key)}
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.06 }}
                  whileHover={{ y: -5 }}
                  whileTap={{ scale: 0.97 }}
                  className={`card text-left overflow-hidden ring-2 transition ${
                    isActive ? "ring-ink/80" : "ring-transparent"
                  }`}
                >
                  <div className={`h-2 ${BAR[c.accent]}`} />
                  <div className="p-5">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-extrabold uppercase tracking-wide text-ink/55">
                        {c.label}
                      </span>
                      <span className="text-2xl">{c.emoji}</span>
                    </div>
                    <div className="font-display text-4xl font-700 text-ink mt-1">{count}</div>
                    <div className="text-sm text-ink/45 mt-1">SKUs · tap to view</div>
                  </div>
                </motion.button>
              )
            })}
          </div>

          <div className="card overflow-hidden">
            <div className="flex items-center gap-3 px-6 pt-5">
              <span className="text-2xl">{cat.emoji}</span>
              <div>
                <div className={`font-display text-xl font-600 ${TEXT[cat.accent]}`}>
                  {cat.label}
                  <span className="text-ink/40 font-sans text-base font-normal"> · {rows.length} SKUs · {scopeLabel}</span>
                </div>
                <div className="text-sm text-ink/55">{cat.hint}</div>
              </div>
            </div>

            <div className="mt-3 max-h-[460px] overflow-y-auto">
              <table className="w-full text-base">
                <thead className="sticky top-0 bg-white/95 backdrop-blur z-10">
                  <tr className="text-left text-ink/50 border-b border-ink/10">
                    <th className="py-3 px-6 font-bold">SKU</th>
                    <th className="py-3 px-3 font-bold">Name</th>
                    <th className="py-3 px-3 font-bold text-right">On hand</th>
                    <th className="py-3 px-6 font-bold text-right">Days of cover</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-12 text-center text-ink/40 text-lg">
                        Nothing in this bucket {storeId ? "for this store" : ""} 🎉
                      </td>
                    </tr>
                  )}
                  {rows.map((r) => (
                    <tr key={r.sku_id} className="border-b border-ink/5 hover:bg-cream/60">
                      <td className="py-2.5 px-6 font-mono text-sm">{r.sku_id}</td>
                      <td className="py-2.5 px-3 text-ink/70">{r.name || "—"}</td>
                      <td className="py-2.5 px-3 text-right">{num(r.inventory)}</td>
                      <td className={`py-2.5 px-6 text-right font-bold ${TEXT[cat.accent]}`}>
                        {r.days_of_cover >= 999 ? "∞" : num(r.days_of_cover)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      <Glossary items={GLOSSARY} />
    </div>
  )
}
