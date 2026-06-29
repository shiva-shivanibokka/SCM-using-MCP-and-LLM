import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Brain, ChevronDown, Send, Trash2 } from "lucide-react"
import { useChat } from "../hooks/useChat"
import LlmSelector from "../components/LlmSelector"
import PageHeader from "../components/PageHeader"
import Markdown from "../components/Markdown"

const SUGGESTIONS = [
  "Which SKUs are at stockout risk this week?",
  "Forecast demand for our top food SKU for 30 days",
  "Rank suppliers by on-time delivery",
  "Where is our revenue concentrated by region?",
]

function ChainOfThought({ steps }) {
  const [open, setOpen] = useState(false)
  if (!steps?.length) return null
  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="pill bg-grape/10 text-grape hover:bg-grape/20 transition"
      >
        <Brain size={14} /> Chain of thought · {steps.length} step
        {steps.length > 1 ? "s" : ""}
        <ChevronDown
          size={14}
          className={`transition ${open ? "rotate-180" : ""}`}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden mt-2 space-y-1 border-l-2 border-grape/30 pl-3"
          >
            {steps.map((s, i) => (
              <li key={i} className="text-xs text-ink/70 break-words">
                <Markdown>{s.content}</Markdown>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function AIAssistant() {
  const { messages, send, connected, busy, reset } = useChat()
  const [input, setInput] = useState("")
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

  const onSend = (text) => {
    const msg = (text ?? input).trim()
    if (!msg || busy) return
    send(msg)
    setInput("")
  }

  return (
    <div>
      <PageHeader
        emoji="🐕‍🦺"
        title="AI Assistant"
        blurb="A ReAct agent that reasons over live data through the Model Context Protocol — it calls real tools (inventory, forecasts, suppliers) and shows its work. Pick a provider, paste a key, and ask anything."
      >
        <span
          className={`pill ${connected ? "bg-leaf/15 text-leaf" : "bg-coral/15 text-coral"}`}
        >
          {connected ? "● Connected" : "● Offline"}
        </span>
      </PageHeader>

      {/* Credentials on one line, on top */}
      <LlmSelector />

      {/* Assistant below */}
      <div className="card mt-4 flex flex-col h-[62vh]">
        <div className="flex items-center justify-between px-4 py-3 border-b border-ink/10">
          <div className="font-display font-600 text-ink">Chat</div>
          <button
            onClick={reset}
            disabled={!messages.length}
            className="pill bg-ink/5 text-ink/60 hover:bg-ink/10 disabled:opacity-40 transition"
          >
            <Trash2 size={13} /> Clear chat
          </button>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="h-full grid place-items-center text-center">
              <div>
                <div className="text-5xl animate-bob">🐾</div>
                <p className="text-ink/50 mt-3 mb-4">
                  Ask me about inventory, forecasts, suppliers, or analytics.
                </p>
                <div className="flex flex-wrap justify-center gap-2 max-w-lg">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => onSend(s)}
                      className="pill bg-teal/10 text-teal hover:bg-teal/20 transition"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="flex justify-end">
                <div className="bg-ink text-white rounded-2xl rounded-br-sm px-4 py-2 text-sm max-w-[80%] whitespace-pre-wrap">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={i} className="flex justify-start">
                <div className="max-w-[85%]">
                  <ChainOfThought steps={m.steps} />
                  {m.error ? (
                    <div className="bg-coral/10 text-coral rounded-2xl rounded-bl-sm px-4 py-3 text-sm whitespace-pre-wrap">
                      ⚠️ {m.error}
                    </div>
                  ) : m.answer ? (
                    <div className="bg-cream border border-ink/10 rounded-2xl rounded-bl-sm px-4 py-3 text-sm text-ink">
                      <Markdown>{m.answer}</Markdown>
                    </div>
                  ) : (
                    <div className="bg-cream border border-ink/10 rounded-2xl px-4 py-3 text-sm text-ink/40">
                      <span className="inline-flex gap-1">
                        <span className="animate-bounce">🐾</span>
                        <span className="animate-bounce [animation-delay:0.15s]">🐾</span>
                        <span className="animate-bounce [animation-delay:0.3s]">🐾</span>
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )
          )}
        </div>

        <div className="flex gap-2 p-3 border-t border-ink/10">
          <input
            className="flex-1 rounded-xl border border-ink/15 px-4 py-2.5 text-sm"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSend()}
            placeholder="Ask about inventory, forecasts, suppliers…"
          />
          <button
            className="bg-teal text-white font-bold px-5 rounded-xl flex items-center gap-2 disabled:opacity-50 hover:brightness-95 transition"
            onClick={() => onSend()}
            disabled={busy || !input.trim()}
          >
            <Send size={16} /> Send
          </button>
        </div>
      </div>
    </div>
  )
}
