import { useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { Eraser, KeyRound } from "lucide-react"
import { apiGet } from "../lib/api"
import { useLLMStore } from "../stores/llmStore"

// One-line model bar: provider · model · API key · clear.
export default function LlmSelector() {
  const { provider, model, apiKey, setProvider, setModel, setApiKey, clearApiKey } =
    useLLMStore()
  const { data: providers } = useQuery({
    queryKey: ["providers"],
    queryFn: () => apiGet("/api/chat/providers"),
  })

  const models = providers?.[provider]?.models ?? []
  const onProvider = (p) => setProvider(p, providers?.[p]?.default_model)

  // Self-heal: if a persisted model doesn't belong to the current provider
  // (e.g. an old Groq model left selected under OpenAI), reset to the default.
  useEffect(() => {
    if (providers && models.length && !models.includes(model)) {
      setModel(providers[provider]?.default_model || models[0])
    }
  }, [providers, provider, model, models, setModel])

  const FREE = { groq: "free", gemini: "free" }

  return (
    <div className="card p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="pill bg-grape/10 text-grape">
          <KeyRound size={13} /> Bring your own key
        </span>

        <select
          className="rounded-xl border border-ink/15 bg-cream px-3 py-2 text-sm font-bold"
          value={provider}
          onChange={(e) => onProvider(e.target.value)}
          aria-label="Provider"
        >
          {providers &&
            Object.keys(providers).map((p) => (
              <option key={p} value={p}>
                {p}
                {FREE[p] ? " (free)" : ""}
              </option>
            ))}
        </select>

        <select
          className="rounded-xl border border-ink/15 bg-cream px-3 py-2 text-sm"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          aria-label="Model"
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>

        <input
          type="password"
          className="flex-1 min-w-[180px] rounded-xl border border-ink/15 px-3 py-2 text-sm"
          placeholder={`Paste your ${provider} API key — stays in your browser`}
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          aria-label="API key"
        />

        <button
          onClick={clearApiKey}
          disabled={!apiKey}
          className="pill bg-coral/10 text-coral disabled:opacity-40 hover:bg-coral/20 transition"
          title="Clear the API key"
        >
          <Eraser size={14} /> Clear
        </button>
      </div>
      <p className="text-[11px] text-ink/45 mt-2 px-1">
        Groq & Gemini have free tiers. Keys never leave your browser — they're sent only
        with your chat request and never stored on the server.
      </p>
    </div>
  )
}
