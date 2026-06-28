import { useEffect, useRef, useState, useCallback } from "react"
import { API_BASE } from "../lib/api"
import { useLLMStore } from "../stores/llmStore"

export function useChat() {
  const wsRef = useRef(null)
  const [messages, setMessages] = useState([])
  const [connected, setConnected] = useState(false)
  const { provider, model, apiKey } = useLLMStore()

  useEffect(() => {
    const url = API_BASE.replace(/^http/, "ws") + "/ws/chat"
    const ws = new WebSocket(url)
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (ev) => {
      const frame = JSON.parse(ev.data)
      setMessages((m) => [...m, { role: "assistant", ...frame }])
    }
    return () => ws.close()
  }, [])

  const send = useCallback(
    (message) => {
      setMessages((m) => [...m, { role: "user", type: "user", content: message }])
      wsRef.current?.send(
        JSON.stringify({ message, provider, model, api_key: apiKey })
      )
    },
    [provider, model, apiKey]
  )

  return { messages, send, connected }
}
