import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from "recharts"
import { apiGet } from "../lib/api"
import PageHeader from "../components/PageHeader"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"
import Glossary from "../components/Glossary"
import { inrCompact, inr } from "../lib/format"

const GLOSSARY = [
  { term: "LTV (Lifetime Value)", what: "The total revenue an average customer in a segment brings over their whole relationship with us — not a single purchase. High-LTV segments are worth retaining hardest." },
  { term: "Customer segment", what: "A behavioural group customers are bucketed into based on what and how they buy." },
  { term: "Loyal Premium", what: "Frequent, high-spend regulars who buy premium brands — the most valuable group." },
  { term: "Multi-Pet Household", what: "Customers with several pets — larger, more frequent baskets across multiple pet types." },
  { term: "Health Focused", what: "Customers who prioritise vet diets, supplements, and wellness products." },
  { term: "New Pet Parent", what: "Recently acquired customers still building their basket — high growth potential." },
  { term: "Budget Conscious", what: "Price-sensitive shoppers who favour value packs and discounts." },
  { term: "Returns by category", what: "How many returns each product category generated — high bars flag quality, sizing, or expectation problems." },
]

const PALETTE = ["#12B5A6", "#FF7A45", "#FF5DA2", "#3DA5F4", "#36C26B", "#8B5CF6", "#FFC53D"]

const VIEWS = [
  { key: "segments", label: "Customer segments", emoji: "🧑‍🤝‍🧑",
    hint: "How many customers fall into each loyalty/value segment." },
  { key: "ltv", label: "Lifetime value", emoji: "💎",
    hint: "Average lifetime value per segment — who's worth retaining." },
  { key: "returns", label: "Returns by category", emoji: "↩️",
    hint: "Where returns concentrate — high bars are quality or sizing risks." },
]

export default function Analytics() {
  const { data } = useQuery({
    queryKey: ["analytics"],
    queryFn: () => apiGet("/api/analytics/overview"),
  })
  const [view, setView] = useState("segments")

  const segments = Object.entries(data?.segments ?? {})
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
  const ltv = Object.entries(data?.ltv_by_segment ?? {})
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
  const returns = Object.entries(data?.return_rate_by_category ?? {})
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)

  const current = VIEWS.find((v) => v.key === view)
  const rows = view === "segments" ? segments : view === "ltv" ? ltv : returns
  const isMoney = view === "ltv"

  return (
    <div>
      <PageHeader
        emoji="🧁"
        title="Customer Analytics"
        blurb="Who shops with us, what they're worth, and where products come back. Switch metrics with the toggles — each renders its own clean chart."
      />

      {ltv.length > 0 && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {ltv.slice(0, 4).map((d, i) => (
            <KpiCard
              key={d.name}
              index={i}
              title={`LTV · ${d.name}`}
              value={inrCompact(d.value)}
              accent={["teal", "pink", "sky", "grape"][i % 4]}
              emoji="💎"
              help={`Average lifetime value of a "${d.name}" customer — total revenue they bring over their whole relationship with us.`}
            />
          ))}
        </div>
      )}

      {/* Metric switcher */}
      <div className="flex flex-wrap gap-2 mb-4">
        {VIEWS.map((v) => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            className={`pill transition ${
              view === v.key
                ? "bg-ink text-white"
                : "bg-white text-ink/65 border border-ink/10 hover:bg-cream"
            }`}
          >
            <span>{v.emoji}</span> {v.label}
          </button>
        ))}
      </div>

      <ChartCard title={current.label} hint={current.hint}>
        {rows.length === 0 ? (
          <div className="text-ink/40 text-sm py-10 text-center">No data available.</div>
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(320, rows.length * 38)}>
            <BarChart data={rows} layout="vertical" margin={{ left: 24, right: 32 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A214010" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12 }}
                tickFormatter={isMoney ? (v) => inrCompact(v) : undefined} />
              <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(v) => (isMoney ? inr(v) : v.toLocaleString("en-IN"))}
                cursor={{ fill: "#2A214008" }}
              />
              <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                {rows.map((_, i) => (
                  <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      <Glossary items={GLOSSARY} />
    </div>
  )
}
