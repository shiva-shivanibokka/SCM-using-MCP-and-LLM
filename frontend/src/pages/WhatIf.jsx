import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"
import Glossary from "../components/Glossary"
import { inr, inrCompact, num } from "../lib/format"

const GLOSSARY = [
  { term: "Price elasticity", what: "How much demand moves when price moves. Estimated from your own discount history — a deeper discount usually lifts units, and elasticity says by how much." },
  { term: "GMV vs Net revenue", what: "GMV is sales at full sticker price; Net revenue is what's left after the discount. A discount lifts units but shrinks the per-unit take — this shows the net effect." },
  { term: "Days of cover", what: "How many days current stock lasts at today's sales rate. Restocking raises it; too high (>90d) means cash tied up in overstock." },
  { term: "Restock ROI", what: "Gross profit from the restocked units divided by what they cost to buy. Higher is a better use of working capital." },
]

const selectCls = "px-3 py-2 rounded-xl border border-ink/15 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-teal/40"

export default function WhatIf() {
  const [tab, setTab] = useState("discount")
  // input state
  const [discount, setDiscount] = useState(20)
  const [category, setCategory] = useState("")
  const [sku, setSku] = useState("")
  const [units, setUnits] = useState(500)
  // "applied" state — only set when the Simulate button is pressed
  const [applied, setApplied] = useState(null)
  const [appliedR, setAppliedR] = useState(null)

  const { data: opts } = useQuery({
    queryKey: ["intel-options"],
    queryFn: () => apiGet("/api/intelligence/options"),
  })

  const disc = useQuery({
    queryKey: ["whatif-discount", applied],
    queryFn: () => apiGet(`/api/intelligence/whatif/discount?new_discount_pct=${applied.discount}${applied.category ? `&category=${encodeURIComponent(applied.category)}` : ""}`),
    enabled: tab === "discount" && !!applied,
  })
  const rest = useQuery({
    queryKey: ["whatif-restock", appliedR],
    queryFn: () => apiGet(`/api/intelligence/whatif/restock?sku_id=${encodeURIComponent(appliedR.sku)}&restock_units=${appliedR.units}`),
    enabled: tab === "restock" && !!appliedR,
  })

  const d = disc.data
  const r = rest.data

  return (
    <div>
      <PageHeader
        emoji="🧪"
        title="What-If Simulator"
        blurb="Test a decision before you make it. Model how a discount would move revenue, or whether restocking a SKU pays off — grounded in your own sales history."
      />

      <div className="flex gap-2 mb-6">
        {[["discount", "💸 Discount impact"], ["restock", "📦 Restock ROI"]].map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)}
            className={`pill transition ${tab === k ? "bg-ink text-white" : "bg-white text-ink/65 border border-ink/10 hover:bg-cream"}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === "discount" ? (
        <>
          <ChartCard title="Discount scenario" hint="Pick a discount level and category, then run the simulation.">
            <div className="flex flex-wrap items-end gap-6 py-2">
              <label className="flex flex-col gap-1">
                <span className="text-sm font-bold text-ink/60">New discount: {discount}%</span>
                <input type="range" min="0" max="60" value={discount}
                  onChange={(e) => setDiscount(+e.target.value)} className="w-64 accent-teal" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm font-bold text-ink/60">Category</span>
                <select value={category} onChange={(e) => setCategory(e.target.value)} className={selectCls + " w-56"}>
                  <option value="">All categories</option>
                  {(opts?.categories ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <button onClick={() => setApplied({ discount, category })}
                className="pill bg-ink text-white hover:bg-ink/90">
                ▶ Simulate impact
              </button>
            </div>
          </ChartCard>

          {!applied ? (
            <div className="text-ink/40 text-sm py-10 text-center">Set a discount and press <b>Simulate impact</b>.</div>
          ) : disc.isFetching ? (
            <div className="text-ink/40 text-sm py-10 text-center">Simulating…</div>
          ) : d && !d.error ? (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 my-6">
                <KpiCard index={0} title="Est. elasticity" value={d.elasticity} accent="grape" emoji="📈"
                  help="Estimated from your discount history. More negative = demand reacts more strongly to price." />
                <KpiCard index={1} title="Units change" value={`${d.delta.units_pct > 0 ? "+" : ""}${d.delta.units_pct}%`}
                  accent="teal" emoji="📦" help="Projected change in units sold at the new discount." />
                <KpiCard index={2} title="Net revenue change"
                  value={`${d.delta.net_revenue_pct > 0 ? "+" : ""}${d.delta.net_revenue_pct}%`}
                  accent={d.delta.net_revenue >= 0 ? "teal" : "coral"} emoji="💰"
                  help="The bottom-line effect after more units but a deeper discount." />
                <KpiCard index={3} title="Net revenue Δ" value={inrCompact(d.delta.net_revenue)}
                  accent={d.delta.net_revenue >= 0 ? "sky" : "coral"} emoji="₹"
                  help="Absolute change in net revenue versus the historical baseline." />
              </div>
              <ChartCard title={`Baseline vs projected — ${d.category}`} hint={d.note}>
                <table className="w-full text-sm">
                  <thead><tr className="text-left text-ink/50 border-b border-ink/10">
                    <th className="py-2 font-bold">Metric</th>
                    <th className="py-2 font-bold text-right">Baseline ({d.baseline.avg_discount_pct}% disc)</th>
                    <th className="py-2 font-bold text-right">At {d.projected.discount_pct}%</th>
                  </tr></thead>
                  <tbody>
                    <tr className="border-b border-ink/5"><td className="py-2">Units</td>
                      <td className="py-2 text-right tabular-nums">{num(d.baseline.units)}</td>
                      <td className="py-2 text-right tabular-nums">{num(d.projected.units)}</td></tr>
                    <tr className="border-b border-ink/5"><td className="py-2">GMV</td>
                      <td className="py-2 text-right tabular-nums">{inr(d.baseline.gmv)}</td>
                      <td className="py-2 text-right tabular-nums">{inr(d.projected.gmv)}</td></tr>
                    <tr><td className="py-2">Net revenue</td>
                      <td className="py-2 text-right tabular-nums">{inr(d.baseline.net_revenue)}</td>
                      <td className="py-2 text-right tabular-nums font-semibold">{inr(d.projected.net_revenue)}</td></tr>
                  </tbody>
                </table>
              </ChartCard>
            </>
          ) : d && d.error ? (
            <div className="text-coral text-sm py-10 text-center">{d.error}</div>
          ) : null}
        </>
      ) : (
        <>
          <ChartCard title="Restock scenario" hint="Pick a SKU and how many units to add, then run the simulation.">
            <div className="flex flex-wrap items-end gap-6 py-2">
              <label className="flex flex-col gap-1">
                <span className="text-sm font-bold text-ink/60">SKU</span>
                <select value={sku} onChange={(e) => setSku(e.target.value)} className={selectCls + " w-72"}>
                  <option value="">Select a SKU…</option>
                  {(opts?.skus ?? []).map((s) => (
                    <option key={s.sku_id} value={s.sku_id}>{s.name} ({s.sku_id})</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm font-bold text-ink/60">Restock units</span>
                <input type="number" value={units} onChange={(e) => setUnits(+e.target.value)}
                  className={selectCls + " w-40"} />
              </label>
              <button onClick={() => sku && setAppliedR({ sku, units })} disabled={!sku}
                className="pill bg-ink text-white hover:bg-ink/90 disabled:opacity-40">
                ▶ Simulate restock
              </button>
            </div>
          </ChartCard>

          {!appliedR ? (
            <div className="text-ink/40 text-sm py-10 text-center">Pick a SKU and press <b>Simulate restock</b>.</div>
          ) : rest.isFetching ? (
            <div className="text-ink/40 text-sm py-10 text-center">Simulating…</div>
          ) : r && r.error ? (
            <div className="text-coral text-sm py-10 text-center">{r.error}</div>
          ) : r ? (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 my-6">
                <KpiCard index={0} title="Days of cover"
                  value={`${r.current.days_of_cover ?? "∞"} → ${r.projected.days_of_cover ?? "∞"}`}
                  accent="teal" emoji="📅" help="Stock cover now versus after the restock." />
                <KpiCard index={1} title="Restock cost" value={inrCompact(r.economics.restock_cost)}
                  accent="pink" emoji="🧾" help="What buying these units costs." />
                <KpiCard index={2} title="Gross profit" value={inrCompact(r.economics.gross_profit)}
                  accent="sky" emoji="💰" help="Profit if the restocked units sell through." />
                <KpiCard index={3} title="ROI" value={`${r.economics.roi_pct}%`}
                  accent="grape" emoji="📈" help="Gross profit ÷ restock cost." />
              </div>
              {r.projected.overstock_risk && (
                <div className="mb-6 px-4 py-3 rounded-xl bg-[#FFC53D]/15 text-[#B4820A] text-sm font-semibold">
                  ⚠️ This restock pushes cover past 90 days — overstock risk (cash tied up).
                </div>
              )}
            </>
          ) : null}
        </>
      )}

      <Glossary items={GLOSSARY} />
    </div>
  )
}
