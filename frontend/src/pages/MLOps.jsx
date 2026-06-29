import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from "recharts"
import { RefreshCw, Loader2, CheckCircle2 } from "lucide-react"
import { apiGet, apiPost } from "../lib/api"
import PageHeader from "../components/PageHeader"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"
import Glossary from "../components/Glossary"

const GLOSSARY = [
  { term: "Ensemble weights", what: "How much each model contributes to the blended forecast. Frozen between fine-tunes so predictions stay reproducible." },
  { term: "Zero-shot (between fine-tunes)", what: "The models keep forecasting new data without retraining; we only retrain on demand or schedule, not every request." },
  { term: "sMAPE", what: "Symmetric Mean Absolute Percentage Error from backtesting — average % the forecast was off, bounded so demand zero-days don't blow it up. Lower is better." },
  { term: "Backtest", what: "We hide the most recent 30 days, retrain on the rest, forecast those 30 days, and compare to what actually happened. That gap is the score." },
  { term: "Model registry", what: "A logbook of every fine-tune run — its version, accuracy, rows trained, and when. So you can see the model improving over time." },
  { term: "Trigger fine-tune", what: "Actually retrains CatBoost on the latest demand for the busiest SKUs, scores it, and appends a new version to the registry." },
  { term: "Run telemetry (receipt)", what: "Every AI Assistant turn is logged with the tools it called, how long each took, status, and an estimated cost — so agent behaviour is observable, not a black box." },
  { term: "Estimated cost", what: "A ballpark from tokens × a rough per-provider rate. It's an estimate (the agent doesn't expose exact billing), so treat it as directional, not an invoice." },
]

const WEIGHT_COLORS = { chronos: "#12B5A6", nhits: "#8B5CF6", catboost: "#FF7A45" }

function fmtTime(iso) {
  if (!iso) return "—"
  const d = new Date(iso)
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
}

export default function MLOps() {
  const qc = useQueryClient()
  const { data } = useQuery({
    queryKey: ["registry"],
    queryFn: () => apiGet("/api/mlops/registry"),
  })
  const finetune = useMutation({
    mutationFn: () => apiPost("/api/mlops/finetune"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["registry"] }),
  })
  const { data: agent } = useQuery({
    queryKey: ["agent-runs"],
    queryFn: () => apiGet("/api/mlops/agent-runs?limit=25"),
    refetchInterval: 15000,
  })

  if (!data) return <div className="text-ink/50 animate-pulse">Loading registry…</div>

  const weights = Object.entries(data.weights).map(([name, w]) => ({
    name,
    pct: Math.round(w * 100),
  }))
  const mape = data.models.map((m) => ({ name: m.name, mape: m.backtest_mape, type: m.type }))
  const history = data.history || []
  const runs = agent?.runs || []
  const STATUS_STYLE = {
    ok: "bg-leaf/15 text-leaf",
    error: "bg-coral/15 text-coral",
    incomplete: "bg-amber/15 text-amber",
  }

  return (
    <div>
      <PageHeader
        emoji="⚙️"
        title="MLOps"
        blurb="Chronos stays zero-shot; CatBoost retrains on demand. Hit Trigger fine-tune to actually retrain CatBoost on the latest demand, backtest it, and log a new version to the model registry below."
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
          {finetune.isPending ? "Retraining…" : "Trigger fine-tune"}
        </button>
      </PageHeader>

      {finetune.isSuccess && (
        <div className="card p-4 mb-6 border-l-4 border-leaf flex items-start gap-3">
          <CheckCircle2 className="text-leaf shrink-0 mt-0.5" size={20} />
          <div className="text-sm">
            <div className="font-bold text-ink">
              Retrain complete · version{" "}
              <span className="font-mono">v{finetune.data.version}</span>
              {finetune.data.backtest_smape != null && (
                <> · sMAPE <span className="font-mono">{finetune.data.backtest_smape}%</span></>
              )}
            </div>
            <div className="text-ink/60 mt-0.5">{finetune.data.message}</div>
            <div className="text-ink/50 mt-1 text-xs">
              Logged to the model registry at {fmtTime(finetune.data.trained_at)}.
            </div>
          </div>
        </div>
      )}
      {finetune.isError && (
        <div className="card p-4 mb-6 border-l-4 border-coral text-sm text-coral">
          Retrain failed: {String(finetune.error?.message || "couldn't reach the backend")}.
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-6">
        <KpiCard index={0} title="Last fine-tune" value={data.last_finetune} accent="teal" emoji="✅"
          help="When CatBoost was last retrained — updates the moment you trigger a run." />
        <KpiCard index={1} title="Next fine-tune" value={data.next_finetune} accent="amber" emoji="📆"
          help="The next scheduled retrain date (90 days after the last run)." />
        <KpiCard index={2} title="Registry versions" value={history.length} accent="grape" emoji="📚"
          help="How many fine-tune runs are logged in the model registry." />
        <KpiCard index={3} title="Models in ensemble" value={data.models.length} accent="sky" emoji="🧩"
          help="Models blended into the forecast: Chronos, N-HiTS, CatBoost." />
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
          title="Backtest error % (lower is better)"
          hint="CatBoost shows the sMAPE from your latest real retrain; Chronos and N-HiTS show reference backtests. Triggering a fine-tune pushes CatBoost's bar to its freshly measured score."
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

      {/* Model registry logbook */}
      <div className="card mt-6 overflow-hidden">
        <div className="px-5 py-4 border-b border-ink/10">
          <div className="font-display font-700 text-lg text-ink">Model registry · fine-tune logbook</div>
          <div className="text-sm text-ink/55 mt-0.5">
            Every retrain appends a version here, so you can track accuracy over time.
          </div>
        </div>
        {history.length === 0 ? (
          <div className="p-8 text-center text-ink/50">
            No fine-tune runs yet. Hit <b>Trigger fine-tune</b> to create version 1.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-ink/55 border-b border-ink/10">
                  <th className="px-5 py-2.5 font-bold">Version</th>
                  <th className="px-5 py-2.5 font-bold">Model</th>
                  <th className="px-5 py-2.5 font-bold">Backtest sMAPE</th>
                  <th className="px-5 py-2.5 font-bold">SKUs</th>
                  <th className="px-5 py-2.5 font-bold">Rows trained</th>
                  <th className="px-5 py-2.5 font-bold">Trained at</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr key={h.version} className="border-b border-ink/5 hover:bg-ink/[0.02]">
                    <td className="px-5 py-2.5 font-mono font-bold text-grape">v{h.version}</td>
                    <td className="px-5 py-2.5">{h.model_name}</td>
                    <td className="px-5 py-2.5 font-mono">
                      {h.backtest_smape != null ? `${h.backtest_smape}%` : "—"}
                    </td>
                    <td className="px-5 py-2.5">{h.n_skus}</td>
                    <td className="px-5 py-2.5">{h.training_rows?.toLocaleString()}</td>
                    <td className="px-5 py-2.5 text-ink/60">{fmtTime(h.trained_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Agent run telemetry — a 'receipt' per assistant turn */}
      <div className="card mt-6 overflow-hidden">
        <div className="px-5 py-4 border-b border-ink/10">
          <div className="font-display font-700 text-lg text-ink">Agent activity · run telemetry</div>
          <div className="text-sm text-ink/55 mt-0.5">
            Every AI Assistant turn is traced here: which tools it called, how long
            each took, status, and an estimated token cost. Auto-refreshes every 15s.
          </div>
        </div>
        {runs.length === 0 ? (
          <div className="p-8 text-center text-ink/50">
            No agent runs yet. Ask something on the <b>AI Assistant</b> tab and it appears here.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-ink/55 border-b border-ink/10">
                  <th className="px-5 py-2.5 font-bold">When</th>
                  <th className="px-5 py-2.5 font-bold">Provider / model</th>
                  <th className="px-5 py-2.5 font-bold">Question</th>
                  <th className="px-5 py-2.5 font-bold">Tools</th>
                  <th className="px-5 py-2.5 font-bold">Latency</th>
                  <th className="px-5 py-2.5 font-bold">Est. tokens</th>
                  <th className="px-5 py-2.5 font-bold">Est. cost</th>
                  <th className="px-5 py-2.5 font-bold">Status</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} className="border-b border-ink/5 align-top hover:bg-ink/[0.02]">
                    <td className="px-5 py-2.5 text-ink/60 whitespace-nowrap">{fmtTime(r.created_at)}</td>
                    <td className="px-5 py-2.5 whitespace-nowrap">
                      <span className="font-bold">{r.provider}</span>
                      <span className="text-ink/45 text-xs block">{r.model || "—"}</span>
                    </td>
                    <td className="px-5 py-2.5 max-w-[280px] truncate" title={r.question}>{r.question}</td>
                    <td className="px-5 py-2.5">
                      {r.steps?.length ? (
                        <div className="flex flex-col gap-0.5">
                          {r.steps.map((s, i) => (
                            <span key={i} className="font-mono text-xs text-ink/70 whitespace-nowrap">
                              {s.tool} <span className="text-ink/40">· {s.ms}ms</span>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-ink/40">{r.n_tools || 0}</span>
                      )}
                    </td>
                    <td className="px-5 py-2.5 font-mono whitespace-nowrap">{(r.latency_ms / 1000).toFixed(1)}s</td>
                    <td className="px-5 py-2.5 font-mono text-ink/70">{r.est_tokens?.toLocaleString()}</td>
                    <td className="px-5 py-2.5 font-mono text-ink/70">${(r.est_cost_usd ?? 0).toFixed(5)}</td>
                    <td className="px-5 py-2.5">
                      <span className={`pill ${STATUS_STYLE[r.status] || "bg-ink/10 text-ink/60"}`}>
                        {r.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <Glossary items={GLOSSARY} />
    </div>
  )
}
