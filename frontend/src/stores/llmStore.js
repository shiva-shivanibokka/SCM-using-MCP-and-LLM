import { create } from "zustand"
import { persist } from "zustand/middleware"

export const useLLMStore = create(
  persist(
    (set) => ({
      provider: "groq",
      model: "llama-3.3-70b-versatile",
      apiKey: "",
      setProvider: (provider) => set({ provider }),
      setModel: (model) => set({ model }),
      setApiKey: (apiKey) => set({ apiKey }),
    }),
    { name: "llm-settings" }
  )
)
