import { useQuery, useMutation } from "@tanstack/react-query"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from "recharts"
import { RefreshCw, Loader2, CheckCircle2 } from "lucide-react"
import { apiGet, apiPost } from "../lib/api"
import PageHeader from "../components/PageHeader"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"

const WEIGHT_COLORS = { chronos: "#12B5A6", nhits: "#8B5CF6", catboost: "#FF7A45" }

export default function MLOps() {
  const { data } = useQuery({
    queryKey: ["registry"],
    queryFn: () => apiGet("/api/mlops/registry"),
  })
  const finetune = useMutation({ mutationFn: () => apiPost("/api/mlops/finetune") })

  if (!data) return <div className="text-ink/50 animate-pulse">Loading registry…</div>

  const weights = Object.entries(data.weights).map(([name, w]) => ({
    name,
    pct: Math.round(w * 100),
  }))
  const mape = data.models.map((m) => ({ name: m.name, mape: m.backtest_mape, type: m.type }))

  return (
    <div>
      <PageHeader
        emoji="⚙️"
        title="MLOps"
        blurb="The forecasting stack is zero-shot between quarterly fine-tunes: ensemble weights stay frozen so predictions are reproducible, and we retrain on a schedule. Here's the model registry, accuracy, and the controls to trigger the next run."
      >
        <button
          onClick={() => finetune.mutate()}
          disabled={finetune.isPending}
          className="bg-grape text-white font-bold px-5 py-2.5 rounded-xl flex items-center gap-2 disabled:opacity-50 hover:brightness-95 transition shadow-pop"
        >
          {finetune.isPending ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <RefreshCw size={16} />
          )}
          Trigger fine-tune
        </button>
      </PageHeader>

      {finetune.isSuccess && (
        <div className="card p-4 mb-6 border-l-4 border-leaf flex items-start gap-3">
          <CheckCircle2 className="text-leaf shrink-0 mt-0.5" size={20} />
          <div className="text-sm">
            <div className="font-bold text-ink">
              Fine-tune queued · job <span className="font-mono">{finetune.data.job_id}</span>
            </div>
            <div className="text-ink/60 mt-0.5">{finetune.data.message}</div>
            <div className="text-ink/50 mt-1 text-xs">
              Retraining: {finetune.data.models.join(" · ")} · next auto-cycle{" "}
              {finetune.data.next_finetune}
            </div>
          </div>
        </div>
      )}
      {finetune.isError && (
        <div className="card p-4 mb-6 border-l-4 border-coral text-sm text-coral">
          Couldn't reach the fine-tune endpoint. Is the backend running?
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard title="Last fine-tune" value={data.last_finetune} accent="teal" emoji="✅" />
        <KpiCard title="Next fine-tune" value={data.next_finetune} accent="amber" emoji="📆" />
        <KpiCard title="Models in ensemble" value={data.models.length} accent="grape" emoji="🧩" />
        <KpiCard title="Cadence" value="90 days" subtitle="zero-shot between" accent="sky" emoji="🔁" />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <ChartCard
          title="Ensemble weights"
          hint="Frozen contribution of each model to the blended forecast. Chronos leads; CatBoost is the cheap fallback that keeps the API up if the heavy models are unavailable."
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={weights}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A214010" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 13, fontWeight: 700 }} />
              <YAxis unit="%" tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => `${v}%`} cursor={{ fill: "#2A214008" }} />
              <Bar dataKey="pct" radius={[8, 8, 0, 0]}>
                {weights.map((w, i) => (
                  <Cell key={i} fill={WEIGHT_COLORS[w.name] || "#12B5A6"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Backtest accuracy (MAPE — lower is better)"
          hint="Mean absolute percentage error from the last backtest. This is what the fine-tune run aims to push down."
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={mape}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A214010" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 13, fontWeight: 700 }} />
              <YAxis unit="%" tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => `${v}%`} cursor={{ fill: "#2A214008" }} />
              <Bar dataKey="mape" radius={[8, 8, 0, 0]}>
                {mape.map((m, i) => (
                  <Cell key={i} fill={WEIGHT_COLORS[m.name] || "#12B5A6"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  )
}
