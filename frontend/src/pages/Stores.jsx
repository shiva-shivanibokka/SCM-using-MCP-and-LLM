import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { motion } from "framer-motion"
import { Search, MapPin } from "lucide-react"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import { titleize } from "../lib/format"

// fields we surface as headline stats (if present); everything else lands in
// the details grid.
const HEADLINE = [
  ["store_type", "Format", "🏷️"],
  ["store_format", "Format", "🏷️"],
  ["size_sqft", "Size (sqft)", "📐"],
  ["region", "Region", "🗺️"],
  ["opened_date", "Opened", "📅"],
]

const fmt = (v) =>
  typeof v === "number" ? v.toLocaleString("en-IN") : String(v ?? "—")

export default function Stores() {
  const { data } = useQuery({
    queryKey: ["stores"],
    queryFn: () => apiGet("/api/stores/grid"),
  })
  const stores = data?.stores ?? []
  const [q, setQ] = useState("")
  const [pickedId, setPickedId] = useState(null)

  const filtered = useMemo(() => {
    const t = q.toLowerCase()
    return stores.filter(
      (s) =>
        !t ||
        String(s.store_id).toLowerCase().includes(t) ||
        String(s.city || "").toLowerCase().includes(t) ||
        String(s.region || "").toLowerCase().includes(t)
    )
  }, [stores, q])

  const active =
    stores.find((s) => s.store_id === pickedId) || filtered[0] || stores[0]

  const headlines = active
    ? HEADLINE.filter(([k]) => active[k] != null).filter(
        (h, i, arr) => arr.findIndex((x) => x[1] === h[1]) === i
      )
    : []

  const detailEntries = active
    ? Object.entries(active).filter(
        ([k]) => !["store_id", "city"].includes(k) && !HEADLINE.some(([hk]) => hk === k)
      )
    : []

  return (
    <div>
      <PageHeader
        emoji="🏬"
        title="Store Network"
        blurb={`${stores.length} stores across the country. Search or click a store on the left to pull up its full profile on the right.`}
      />

      <div className="grid lg:grid-cols-[340px_1fr] gap-6">
        {/* List */}
        <div className="card p-3 h-[68vh] flex flex-col">
          <div className="relative mb-2">
            <Search size={15} className="absolute left-3 top-2.5 text-ink/40" />
            <input
              className="w-full rounded-xl border border-ink/15 pl-9 pr-3 py-2 text-sm"
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
                  className={`w-full text-left rounded-xl px-3 py-2 transition ${
                    on ? "bg-ink text-white" : "hover:bg-cream"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm">{s.city || s.store_id}</span>
                    <span className={`text-[10px] font-mono ${on ? "text-white/60" : "text-ink/40"}`}>
                      {s.store_id}
                    </span>
                  </div>
                  <div className={`text-xs ${on ? "text-white/70" : "text-ink/50"}`}>
                    {s.region || ""} {s.store_type ? `· ${s.store_type}` : ""}
                  </div>
                </button>
              )
            })}
            {filtered.length === 0 && (
              <div className="text-center text-ink/40 text-sm py-8">No stores match.</div>
            )}
          </div>
        </div>

        {/* Detail */}
        {active && (
          <motion.div
            key={active.store_id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="card p-6">
              <div className="flex items-center gap-4">
                <div className="h-16 w-16 grid place-items-center rounded-2xl bg-orange/10 text-3xl">
                  🏬
                </div>
                <div>
                  <div className="font-display text-3xl font-700 text-ink flex items-center gap-2">
                    {active.city || active.store_id}
                  </div>
                  <div className="text-sm text-ink/55 flex items-center gap-1">
                    <MapPin size={13} /> {active.region || "—"} · {active.store_id}
                  </div>
                </div>
              </div>

              {headlines.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6">
                  {headlines.map(([k, label, emoji]) => (
                    <div key={label} className="rounded-2xl bg-cream p-4">
                      <div className="text-lg">{emoji}</div>
                      <div className="font-display text-xl font-700 text-ink mt-1">
                        {fmt(active[k])}
                      </div>
                      <div className="text-[11px] font-bold uppercase tracking-wide text-ink/50">
                        {label}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {detailEntries.length > 0 && (
              <div className="card p-6">
                <div className="font-display text-lg font-600 text-ink mb-3">
                  Full profile
                </div>
                <div className="grid sm:grid-cols-2 gap-x-8 gap-y-2">
                  {detailEntries.map(([k, v]) => (
                    <div
                      key={k}
                      className="flex justify-between gap-4 border-b border-ink/5 py-1.5 text-sm"
                    >
                      <span className="text-ink/55">{titleize(k)}</span>
                      <span className="font-bold text-ink text-right">{fmt(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  )
}
