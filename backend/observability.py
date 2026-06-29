"""Agent run telemetry — the 'receipt' for every assistant turn.

Each ReAct run is logged to an `agent_runs` table: which provider/model, the
question, which tools were called and how long each took, total latency, status,
and an estimated token/cost figure. The MLOps tab reads this back.

Token cost is an ESTIMATE (chars/4 heuristic × a rough per-provider blended
rate) because the agent's LLM calls don't surface exact usage. It's labelled as
estimated in the UI so it's never mistaken for a billed amount.
"""
from __future__ import annotations

import json
import logging

from .db import get_engine

logger = logging.getLogger(__name__)

# Rough blended USD per 1K tokens (input+output averaged) — for a ballpark only.
_RATES = {"anthropic": 0.006, "openai": 0.005, "groq": 0.0001, "gemini": 0.0003}


def _ensure_table(engine) -> None:
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id       SERIAL PRIMARY KEY,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                provider     TEXT,
                model        TEXT,
                question     TEXT,
                n_tools      INTEGER,
                tools        TEXT,
                latency_ms   INTEGER,
                status       TEXT,
                error        TEXT,
                est_tokens   INTEGER,
                est_cost_usd DOUBLE PRECISION,
                steps        JSONB
            )
        """))


def record_run(*, provider: str, model: str, question: str, tools: list[str],
               steps: list[dict], latency_ms: int, status: str, error: str,
               est_tokens: int) -> None:
    """Persist one agent run. Best-effort: never raises into the chat path."""
    engine = get_engine()
    if engine is None:
        return
    try:
        _ensure_table(engine)
        cost = round(est_tokens / 1000.0 * _RATES.get(provider, 0.003), 6)
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO agent_runs
                    (provider, model, question, n_tools, tools, latency_ms,
                     status, error, est_tokens, est_cost_usd, steps)
                VALUES (:provider, :model, :question, :n_tools, :tools, :latency_ms,
                        :status, :error, :est_tokens, :est_cost_usd, :steps)
            """), {
                "provider": provider, "model": model, "question": question[:500],
                "n_tools": len(tools), "tools": ", ".join(tools),
                "latency_ms": latency_ms, "status": status, "error": (error or "")[:500],
                "est_tokens": est_tokens, "est_cost_usd": cost,
                "steps": json.dumps(steps),
            })
    except Exception as e:  # telemetry must never break the chat
        logger.warning("record_run failed: %s", e)


def recent_runs(limit: int = 25) -> list[dict]:
    engine = get_engine()
    if engine is None:
        return []
    try:
        _ensure_table(engine)
        from sqlalchemy import text
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT run_id, created_at, provider, model, question, n_tools,
                       tools, latency_ms, status, est_tokens, est_cost_usd, steps
                FROM agent_runs ORDER BY run_id DESC LIMIT :l
            """), {"l": limit}).all()
        out = []
        for r in rows:
            steps = r[11]
            if isinstance(steps, str):
                steps = json.loads(steps)
            out.append({
                "run_id": int(r[0]), "created_at": r[1].isoformat(),
                "provider": r[2], "model": r[3], "question": r[4],
                "n_tools": r[5], "tools": r[6], "latency_ms": r[7],
                "status": r[8], "est_tokens": r[9], "est_cost_usd": r[10],
                "steps": steps or [],
            })
        return out
    except Exception as e:
        logger.warning("recent_runs failed: %s", e)
        return []
