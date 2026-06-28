from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_providers_list():
    r = client.get("/api/chat/providers")
    assert r.status_code == 200
    body = r.json()
    assert "groq" in body and "anthropic" in body


def test_ws_echo_done(monkeypatch):
    import backend.agent_ws as aws

    async def fake_stream(message, provider, model, api_key):
        yield {"type": "step", "content": "thinking"}
        yield {"type": "done", "content": "hello " + message}

    monkeypatch.setattr(aws, "stream_agent", fake_stream)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "world", "provider": "groq",
                      "model": "llama-3.1-8b-instant", "api_key": "x"})
        frames = []
        while True:
            f = ws.receive_json()
            frames.append(f)
            if f["type"] == "done":
                break
    assert frames[-1] == {"type": "done", "content": "hello world"}
