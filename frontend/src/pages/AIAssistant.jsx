import { useState } from "react"
import { useChat } from "../hooks/useChat"
import LlmSelector from "../components/LlmSelector"

export default function AIAssistant() {
  const { messages, send, connected } = useChat()
  const [input, setInput] = useState("")

  const onSend = () => {
    if (!input.trim()) return
    send(input)
    setInput("")
  }

  return (
    <div className="flex gap-6 h-[85vh]">
      <div className="w-72 shrink-0">
        <LlmSelector />
      </div>
      <div className="flex-1 flex flex-col bg-white rounded-2xl shadow p-4">
        <div className="text-sm font-semibold text-navy mb-2">
          AI Assistant {connected ? "🟢" : "🔴"}
        </div>
        <div className="flex-1 overflow-y-auto space-y-2 pr-1">
          {messages.length === 0 && (
            <div className="text-navy/40 text-sm">
              Ask anything about inventory, forecasts, suppliers, or analytics.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
              <span
                className={`inline-block px-3 py-2 rounded-2xl text-sm max-w-[80%] whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-teal text-white"
                    : m.type === "error"
                    ? "bg-coral/20 text-coral"
                    : m.type === "step"
                    ? "bg-cream text-navy/60 text-xs"
                    : "bg-cream text-navy"
                }`}
              >
                {m.content}
              </span>
            </div>
          ))}
        </div>
        <div className="flex gap-2 mt-3">
          <input
            className="flex-1 rounded-xl border p-2 text-sm"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSend()}
            placeholder="Ask about inventory, forecasts, suppliers…"
          />
          <button className="bg-teal text-white px-4 rounded-xl" onClick={onSend}>
            Send
          </button>
        </div>
      </div>
    </div>
  )
}
