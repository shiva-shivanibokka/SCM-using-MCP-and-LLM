"""Adapter: existing ReAct agent generator -> normalized frames for WebSocket.

The agent's `run_agent_with_steps` already accepts a per-call `api_key` (UI
key wins over .env) and yields rich dict frames. We normalize those frames to
a compact `{type, content}` contract the frontend renders, then emit a final
`done` frame so the client knows the turn is complete.
"""
from __future__ import annotations

from typing import AsyncGenerator

from agent.agent import run_agent_with_steps


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
    async for frame in run_agent_with_steps(
        message, provider=provider, model=model, api_key=api_key
    ):
        norm = _normalize(frame)
        if norm["type"] == "answer":
            final = norm["content"]
        yield norm
    yield {"type": "done", "content": final}
