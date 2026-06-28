import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"

export default function MLOps() {
  const { data } = useQuery({
    queryKey: ["registry"],
    queryFn: () => apiGet("/api/mlops/registry"),
  })
  if (!data) return <div className="text-navy/50">Loading…</div>
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">MLOps</h1>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <KpiCard title="Last Fine-tune" value={data.last_finetune} accent="teal" />
        <KpiCard title="Next Fine-tune" value={data.next_finetune} accent="amber" />
        <KpiCard title="Models" value={data.models.length} accent="teal" />
      </div>
      <ChartCard title="Ensemble Weights (frozen between fine-tunes — zero-shot)">
        <ul className="text-sm space-y-1">
          {Object.entries(data.weights).map(([k, v]) => (
            <li key={k} className="flex justify-between border-b py-1">
              <span>{k}</span>
              <span className="font-semibold">{(v * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
      </ChartCard>
      <ChartCard title="Backtest Accuracy (MAPE — lower is better)">
        <ul className="text-sm space-y-1">
          {data.models.map((m) => (
            <li key={m.name} className="flex justify-between border-b py-1">
              <span>
                {m.name} <span className="text-navy/40">({m.type})</span>
              </span>
              <span className="font-semibold">{m.backtest_mape}%</span>
            </li>
          ))}
        </ul>
      </ChartCard>
    </div>
  )
}
