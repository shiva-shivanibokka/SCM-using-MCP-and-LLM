import { create } from "zustand"
import { persist } from "zustand/middleware"

// Default model per provider. Used when the provider changes so we never send
// (say) a Groq model name to OpenAI — which returns a 404 "model_not_found".
export const DEFAULT_MODEL = {
  anthropic: "claude-sonnet-4-6",
  openai: "gpt-4o",
  groq: "llama-3.3-70b-versatile",
  gemini: "gemini-2.0-flash",
}

export const useLLMStore = create(
  persist(
    (set) => ({
      provider: "groq",
      model: "llama-3.3-70b-versatile",
      apiKey: "",
      // Switching provider resets the model to that provider's default so the
      // model and provider can never drift out of sync.
      setProvider: (provider, defaultModel) =>
        set({ provider, model: defaultModel || DEFAULT_MODEL[provider] || "" }),
      setModel: (model) => set({ model }),
      setApiKey: (apiKey) => set({ apiKey }),
      clearApiKey: () => set({ apiKey: "" }),
    }),
    { name: "llm-settings" }
  )
)
