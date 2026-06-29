"""Adapter: existing ReAct agent generator -> normalized frames for WebSocket.

The agent's `run_agent_with_steps` already accepts a per-call `api_key` (UI
key wins over .env) and yields rich dict frames. We normalize those frames to
a compact `{type, content}` contract the frontend renders, then emit a final
`done` frame so the client knows the turn is complete.

Along the way we time each tool call and log the whole run to the agent_runs
telemetry table (backend.observability) — the 'receipt' shown in the MLOps tab.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from agent.agent import run_agent_with_steps

from .observability import record_run


def _normalize(frame: dict) -> dict:
    t = frame.get("type")
    if t == "tool_start":
        return {"type": "step", "content": f"🔧 {frame.get('tool')}({frame.get('input')})"}
    if t == "tool_result":
        return {"type": "step", "content": f"↳ {frame.get('result')}"}
    if t == "thinking":
        return {"type": "step", "content": frame.get("text", "")}
    if t == "answer":
        return {"type": "answer", "content": frame.get("text", "")}
    if t == "error":
        return {"type": "error", "content": frame.get("text", "")}
    return {"type": "step", "content": str(frame)}


async def stream_agent(message: str, provider: str, model: str,
                       api_key: str) -> AsyncGenerator[dict, None]:
    final = ""
    t0 = time.perf_counter()
    steps: list[dict] = []
    tools: list[str] = []
    status = "incomplete"
    error_text = ""
    chars = len(message or "")
    pending: tuple[str, float] | None = None  # (tool_name, start_time)

    async for frame in run_agent_with_steps(
        message, provider=provider, model=model, api_key=api_key
    ):
        t = frame.get("type")
        if t == "tool_start":
            pending = (str(frame.get("tool")), time.perf_counter())
        elif t == "tool_result":
            name = str(frame.get("tool"))
            ms = int((time.perf_counter() - (pending[1] if pending else t0)) * 1000)
            steps.append({"kind": "tool", "tool": name, "ms": ms})
            tools.append(name)
            chars += len(str(frame.get("result", "")))
            pending = None
        elif t == "thinking":
            chars += len(frame.get("text", ""))
        elif t == "answer":
            final = frame.get("text", "")
            chars += len(final)
            status = "ok"
        elif t == "error":
            error_text = frame.get("text", "")
            status = "error"

        norm = _normalize(frame)
        if norm["type"] == "answer":
            final = norm["content"]
        yield norm

    latency_ms = int((time.perf_counter() - t0) * 1000)
    # Best-effort telemetry on a worker thread so the DB write never blocks the loop.
    try:
        await asyncio.to_thread(
            record_run,
            provider=provider, model=model, question=message, tools=tools,
            steps=steps, latency_ms=latency_ms, status=status, error=error_text,
            est_tokens=chars // 4,
        )
    except Exception:
        pass

    yield {"type": "done", "content": final}
