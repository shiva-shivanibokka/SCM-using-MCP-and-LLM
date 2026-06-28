import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import { useLLMStore } from "../stores/llmStore"

export default function LlmSelector() {
  const { provider, model, apiKey, setProvider, setModel, setApiKey } = useLLMStore()
  const { data: providers } = useQuery({
    queryKey: ["providers"],
    queryFn: () => apiGet("/api/chat/providers"),
  })

  const models = providers?.[provider]?.models ?? []

  return (
    <div className="flex flex-col gap-2 p-4 bg-white rounded-2xl shadow">
      <div className="text-sm font-semibold text-navy mb-1">Model Settings</div>
      <label className="text-xs font-semibold text-navy/70">Provider</label>
      <select
        className="rounded-lg border p-2 text-sm"
        value={provider}
        onChange={(e) => setProvider(e.target.value)}
      >
        {providers &&
          Object.keys(providers).map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
      </select>
      <label className="text-xs font-semibold text-navy/70">Model</label>
      <select
        className="rounded-lg border p-2 text-sm"
        value={model}
        onChange={(e) => setModel(e.target.value)}
      >
        {models.map((m) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
      <label className="text-xs font-semibold text-navy/70">API Key</label>
      <input
        type="password"
        className="rounded-lg border p-2 text-sm"
        placeholder="paste your key (stays in your browser)"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
      />
      <p className="text-[10px] text-navy/40 mt-1">
        Free options: Groq, Gemini, Ollama. Keys are stored only in your browser.
      </p>
    </div>
  )
}
