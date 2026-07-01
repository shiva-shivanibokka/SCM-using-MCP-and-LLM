import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"
import Glossary from "../components/Glossary"
import { num } from "../lib/format"

const GLOSSARY = [
  { term: "Sales crash", what: "A SKU whose weekly revenue dropped more than 30% versus the week before — a sign of a listing, pricing, or availability problem." },
  { term: "Inventory spike", what: "Stock that jumped more than 50% overnight. Usually a data-entry or goods-receipt error rather than a real delivery." },
  { term: "Discount breach", what: "Sales given a deeper discount than the channel's allowed ceiling — margin leakage or a misconfigured promotion." },
  { term: "Velocity-vs-stock risk", what: "A fast-selling SKU with fewer than 14 days of stock left — an imminent stockout that needs action now." },
  { term: "Severity", what: "Critical = act today · Warning = review this week. Ranked so the most urgent issues sit at the top." },
]

const SEV_STYLE = {
  critical: { chip: "bg-coral/15 text-coral", dot: "🔴" },
  warning: { chip: "bg-[#FF7A45]/15 text-[#FF7A45]", dot: "🟠" },
  info: { chip: "bg-sky/15 text-sky", dot: "🔵" },
}

export default function Anomaly() {
  const { data, isLoading } = useQuery({
    queryKey: ["anomaly"],
    queryFn: () => apiGet("/api/intelligence/anomaly"),
  })
  const anomalies = data?.anomalies ?? []
  const s = data?.summary ?? {}
  const byType = s?.by_type ?? {}

  return (
    <div>
      <PageHeader
        emoji="🚨"
        title="Anomaly Detection"
        blurb="Automatic scan of sales and inventory for things that look wrong — revenue crashes, overnight stock spikes, discount breaches, and imminent stockouts — ranked by urgency."
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard index={0} title="Total anomalies" value={num(s?.total ?? 0)} accent="grape" emoji="🚨"
          help="All issues surfaced across the four detectors." />
        <KpiCard index={1} title="Critical" value={num(s?.critical ?? 0)} accent="coral" emoji="🔴"
          help="Act today — the most severe issues." />
        <KpiCard index={2} title="Warning" value={num(s?.warning ?? 0)} accent="pink" emoji="🟠"
          help="Review this week." />
        <KpiCard index={3} title="Detectors firing"
          value={num(Object.keys(byType).length)} accent="teal" emoji="🔎"
          help="How many of the four detector types found something." />
      </div>

      <ChartCard title="Anomalies — most urgent first"
        hint="Each row is one detected issue with a plain-language explanation.">
        {isLoading ? (
          <div className="text-ink/40 text-sm py-10 text-center">Loading…</div>
        ) : anomalies.length === 0 ? (
          <div className="text-teal text-sm py-10 text-center">✓ No anomalies detected.</div>
        ) : (
          <div className="max-h-[560px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="text-left text-ink/50 border-b border-ink/10">
                  <th className="py-2 pr-3 font-bold">Severity</th>
                  <th className="py-2 pr-3 font-bold">Type</th>
                  <th className="py-2 pr-3 font-bold">Entity</th>
                  <th className="py-2 pr-3 font-bold">What happened</th>
                </tr>
              </thead>
              <tbody>
                {anomalies.map((a, i) => (
                  <tr key={i} className="border-b border-ink/5 hover:bg-cream">
                    <td className="py-2 pr-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-bold capitalize ${(SEV_STYLE[a.severity] ?? {}).chip ?? ""}`}>
                        {(SEV_STYLE[a.severity] ?? {}).dot} {a.severity}
                      </span>
                    </td>
                    <td className="py-2 pr-3 font-semibold text-ink whitespace-nowrap">{a.type}</td>
                    <td className="py-2 pr-3 text-ink/60 tabular-nums whitespace-nowrap">{a.entity}</td>
                    <td className="py-2 pr-3 text-ink/80">{a.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </ChartCard>

      <Glossary items={GLOSSARY} />
    </div>
  )
}
