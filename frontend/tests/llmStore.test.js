import { describe, it, expect, beforeEach } from "vitest"
import { useLLMStore } from "../src/stores/llmStore"

describe("llmStore", () => {
  beforeEach(() => {
    useLLMStore.setState({
      provider: "groq",
      model: "llama-3.3-70b-versatile",
      apiKey: "",
    })
  })
  it("updates provider", () => {
    useLLMStore.getState().setProvider("anthropic")
    expect(useLLMStore.getState().provider).toBe("anthropic")
  })
  it("stores api key", () => {
    useLLMStore.getState().setApiKey("sk-test")
    expect(useLLMStore.getState().apiKey).toBe("sk-test")
  })
})
