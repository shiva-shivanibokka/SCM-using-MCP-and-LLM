from __future__ import annotations
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent.agent import PROVIDERS
import backend.agent_ws as agent_ws

router = APIRouter(tags=["chat"])


@router.get("/api/chat/providers")
def providers():
    return PROVIDERS


@router.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            req = await ws.receive_json()
            async for frame in agent_ws.stream_agent(
                req.get("message", ""), req.get("provider", "groq"),
                req.get("model", ""), req.get("api_key", ""),
            ):
                await ws.send_json(frame)
    except WebSocketDisconnect:
        return
