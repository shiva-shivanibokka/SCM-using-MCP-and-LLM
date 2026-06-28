import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { Search, MapPin } from "lucide-react"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import KpiCard from "../components/KpiCard"
import Glossary from "../components/Glossary"
import { inr, inrCompact, num, titleize } from "../lib/format"

const GLOSSARY = [
  { term: "Revenue", what: "Total net sales rung up at this store over the full history." },
  { term: "Orders", what: "Number of separate transactions (baskets) at this store." },
  { term: "Units sold", what: "Total item quantity sold — a basket can hold several units." },
  { term: "Inventory value", what: "Cost-value of stock currently held at this store." },
  { term: "SKUs stocked", what: "How many distinct products this store carries." },
  { term: "Risk mix", what: "Stock health of this store's SKUs: OK (healthy), Warning (getting low), Critical (about to stock out)." },
  { term: "Store format", what: "Standard / Express / Flagship are physical sizes; Spa adds grooming; App/Online are digital channels." },
]

const RISK_META = {
  OK: { tint: "bg-leaf/15 text-leaf", ring: "ring-leaf", dot: "🟢" },
  WARNING: { tint: "bg-amber/15 text-amber", ring: "ring-amber", dot: "🟡" },
  CRITICAL: { tint: "bg-coral/15 text-coral", ring: "ring-coral", dot: "🔴" },
}

export default function Stores() {
  const { data } = useQuery({ queryKey: ["stores"], queryFn: () => apiGet("/api/stores/grid") })
  const stores = data?.stores ?? []
  const [q, setQ] = useState("")
  const [pickedId, setPickedId] = useState(null)
  const [riskFilter, setRiskFilter] = useState("CRITICAL")

  const filtered = useMemo(() => {
    const t = q.toLowerCase()
    return stores.filter(
      (s) => !t ||
        String(s.store_id).toLowerCase().includes(t) ||
        String(s.city || "").toLowerCase().includes(t) ||
        String(s.region || "").toLowerCase().includes(t)
    )
  }, [stores, q])

  const active = stores.find((s) => s.store_id === pickedId) || filtered[0] || stores[0]

  const { data: detail, isFetching } = useQuery({
    queryKey: ["store-detail", active?.store_id],
    queryFn: () => apiGet(`/api/stores/${active.store_id}`),
    enabled: !!active?.store_id,
  })

  const k = detail?.kpis

  return (
    <div>
      <PageHeader
        emoji="🏬"
        title="Store Network"
        blurb={`${stores.length} stores nationwide. Search or click a store to pull up its full profile — sales, stock health, and what sells there.`}
      />

      <div className="grid lg:grid-cols-[380px_1fr] gap-6 items-start">
        {/* List — sticky so it stays beside the (taller) detail column */}
        <div className="card p-4 h-[calc(100vh-160px)] lg:sticky lg:top-28 flex flex-col">
          <div className="relative mb-3">
            <Search size={17} className="absolute left-3 top-3 text-ink/40" />
            <input
              className="w-full rounded-xl border border-ink/15 pl-10 pr-3 py-2.5 text-base"
              placeholder="Search city, region, ID…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          <div className="overflow-y-auto space-y-1.5 pr-1">
            {filtered.map((s) => {
              const on = active && s.store_id === active.store_id
              return (
                <button
                  key={s.store_id}
                  onClick={() => setPickedId(s.store_id)}
                  className={`w-full text-left rounded-xl px-3 py-2.5 transition ${
                    on ? "bg-ink text-white" : "hover:bg-cream"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold">{s.city || s.store_id}</span>
                    <span className={`text-xs font-mono ${on ? "text-white/60" : "text-ink/40"}`}>
                      {s.store_id}
                    </span>
                  </div>
                  <div className={`text-sm ${on ? "text-white/70" : "text-ink/50"}`}>
                    {s.region || ""} {s.store_type ? `· ${s.store_type}` : ""}
                  </div>
                </button>
              )
            })}
            {filtered.length === 0 && (
              <div className="text-center text-ink/40 py-8">No stores match.</div>
            )}
          </div>
        </div>

        {/* Detail */}
        {active && (
          <motion.div key={active.store_id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
            <div className="card p-6">
              <div className="flex items-center gap-4">
                <div className="h-16 w-16 grid place-items-center rounded-2xl bg-orange/10 text-3xl">🏬</div>
                <div>
                  <div className="font-display text-3xl font-700 text-ink">{active.city || active.store_id}</div>
                  <div className="text-base text-ink/55 flex items-center gap-1">
                    <MapPin size={15} /> {active.region || "—"} · {active.store_type || "—"} · {active.store_id}
                  </div>
                </div>
              </div>
            </div>

            {/* Store KPIs */}
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-5">
              <KpiCard index={0} title="Revenue" value={k ? inrCompact(k.revenue) : "…"} accent="teal" emoji="💰"
                help="Total net sales rung up at this store." />
              <KpiCard index={1} title="Orders" value={k ? num(k.orders) : "…"} accent="sky" emoji="🧾"
                help="Number of transactions (baskets) at this store." />
              <KpiCard index={2} title="Units sold" value={k ? num(k.units) : "…"} accent="pink" emoji="📦"
                help="Total item quantity sold across all orders." />
              <KpiCard index={3} title="Gross margin" value={k ? `${k.gross_margin_pct}%` : "…"} accent="amber" emoji="📈"
                help="Profit share after cost of goods at this store." />
              <KpiCard index={4} title="Inventory value" value={k ? inrCompact(k.inventory_value) : "…"} accent="grape" emoji="🏷️"
                help="Cost-value of stock currently held here." />
              <KpiCard index={5} title="SKUs stocked" value={k ? num(k.sku_count) : "…"} accent="leaf" emoji="🔢"
                help="Distinct products carried at this store." />
            </div>

            {/* Stock health — click a status to list those SKUs */}
            <div className="card p-6">
              <div className="font-display text-2xl font-600 text-ink mb-1">Stock health</div>
              <div className="text-base text-ink/55 mb-4">
                Tap a status to see exactly which SKUs are affected at this store.
              </div>
              <div className="flex flex-wrap gap-3 mb-4">
                {["OK", "WARNING", "CRITICAL"].map((r) => {
                  const on = riskFilter === r
                  const m = RISK_META[r]
                  return (
                    <button
                      key={r}
                      onClick={() => setRiskFilter(r)}
                      className={`pill text-lg px-5 py-2.5 ${m.tint} ring-2 transition ${
                        on ? m.ring : "ring-transparent hover:ring-ink/10"
                      }`}
                    >
                      {m.dot} {r} · {detail?.risk_counts?.[r] ?? 0}
                    </button>
                  )
                })}
              </div>

              {isFetching ? (
                <div className="text-ink/40 animate-pulse text-lg py-8">Loading…</div>
              ) : (
                <div className="max-h-[380px] overflow-y-auto rounded-2xl border border-ink/10">
                  <table className="w-full text-base">
                    <thead className="sticky top-0 bg-white/95 backdrop-blur">
                      <tr className="text-left text-ink/50 border-b border-ink/10">
                        <th className="py-3 px-5 font-bold">SKU</th>
                        <th className="py-3 px-3 font-bold">Name</th>
                        <th className="py-3 px-3 font-bold text-right">On hand</th>
                        <th className="py-3 px-5 font-bold text-right">Days of cover</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(detail?.risk_skus?.[riskFilter] ?? []).length === 0 && (
                        <tr><td colSpan={4} className="py-10 text-center text-ink/40 text-lg">
                          No SKUs at {riskFilter} for this store 🎉
                        </td></tr>
                      )}
                      {(detail?.risk_skus?.[riskFilter] ?? []).map((s) => (
                        <tr key={s.sku_id} className="border-b border-ink/5 hover:bg-cream/60">
                          <td className="py-2.5 px-5 font-mono text-sm">{s.sku_id}</td>
                          <td className="py-2.5 px-3 text-ink/70">{s.name || "—"}</td>
                          <td className="py-2.5 px-3 text-right">{num(s.inventory)}</td>
                          <td className="py-2.5 px-5 text-right font-bold">
                            {s.days_of_cover == null ? "∞" : num(s.days_of_cover)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Top 10 categories */}
            <div className="card p-6">
              <div className="font-display text-2xl font-600 text-ink mb-3">Top 10 categories by revenue</div>
              <ul className="divide-y divide-ink/5">
                {(detail?.top_categories ?? []).map((c, i) => (
                  <li key={c.category} className="flex items-center justify-between py-2.5 text-lg">
                    <span className="flex items-center gap-3">
                      <span className="text-ink/30 font-mono text-base w-6">{i + 1}</span>
                      {c.category}
                    </span>
                    <span className="font-bold">{inr(c.revenue)}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Full profile */}
            <div className="card p-6">
              <div className="font-display text-xl font-600 text-ink mb-3">Full profile</div>
              <div className="grid sm:grid-cols-2 gap-x-10 gap-y-2">
                {Object.entries(active).map(([key, v]) => (
                  <div key={key} className="flex justify-between gap-4 border-b border-ink/5 py-2 text-base">
                    <span className="text-ink/55">{titleize(key)}</span>
                    <span className="font-bold text-ink text-right">{String(v ?? "—")}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </div>

      <Glossary items={GLOSSARY} />
    </div>
  )
}
