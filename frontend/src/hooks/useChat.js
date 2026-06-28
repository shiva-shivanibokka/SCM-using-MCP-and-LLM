import { useEffect, useRef, useState, useCallback } from "react"
import { API_BASE } from "../lib/api"
import { useLLMStore } from "../stores/llmStore"

// Turns:
//   { role: "user", content }
//   { role: "assistant", steps: [{content}], answer, error, done }
// Step frames are the agent's Chain of Thought (tool calls + reasoning); the
// final `answer` frame is the reply. `done` closes the turn.
export function useChat() {
  const wsRef = useRef(null)
  const [messages, setMessages] = useState([])
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const { provider, model, apiKey } = useLLMStore()

  useEffect(() => {
    const url = API_BASE.replace(/^http/, "ws") + "/ws/chat"
    const ws = new WebSocket(url)
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (ev) => {
      const frame = JSON.parse(ev.data)
      setMessages((prev) => {
        const next = [...prev]
        // find the last assistant turn (the open one)
        let i = next.length - 1
        while (i >= 0 && next[i].role !== "assistant") i--
        if (i < 0) return next
        const turn = { ...next[i] }
        if (frame.type === "step") turn.steps = [...turn.steps, { content: frame.content }]
        else if (frame.type === "answer") turn.answer = frame.content
        else if (frame.type === "error") turn.error = frame.content
        else if (frame.type === "done") {
          turn.done = true
          if (!turn.answer && !turn.error) turn.answer = frame.content || ""
        }
        next[i] = turn
        return next
      })
      if (frame.type === "done") setBusy(false)
    }
    return () => ws.close()
  }, [])

  const send = useCallback(
    (message) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      setMessages((m) => [
        ...m,
        { role: "user", content: message },
        { role: "assistant", steps: [], answer: "", error: "", done: false },
      ])
      setBusy(true)
      wsRef.current.send(JSON.stringify({ message, provider, model, api_key: apiKey }))
    },
    [provider, model, apiKey]
  )

  const reset = useCallback(() => setMessages([]), [])

  return { messages, send, connected, busy, reset }
}
