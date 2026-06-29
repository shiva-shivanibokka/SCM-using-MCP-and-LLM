"""Model registry backing the MLOps dashboard.

Static metadata (ensemble weights, reference backtests) comes from defaults /
registry.json. Real fine-tune runs are logged to a `model_registry` table in
Postgres — that table is the logbook the dashboard shows, and each retrain
appends a new version row.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from ..db import get_engine
from .ensemble import ENSEMBLE_WEIGHTS, _NAMES

logger = logging.getLogger(__name__)

_REGISTRY_FILE = Path(__file__).parent / "registry.json"
_FINETUNE_PERIOD_DAYS = 90


def _defaults() -> dict:
    last = date(2025, 12, 1)
    return {
        "last_finetune": last.isoformat(),
        "next_finetune": (last + timedelta(days=_FINETUNE_PERIOD_DAYS)).isoformat(),
        "weights": dict(zip(_NAMES, ENSEMBLE_WEIGHTS)),
        "models": [
            {"name": "chronos", "type": "amazon/chronos-t5-small", "backtest_mape": 12.4},
            {"name": "nhits", "type": "neuralforecast/NHITS", "backtest_mape": 14.1},
            {"name": "catboost", "type": "CatBoost/Quantile", "backtest_mape": 16.8},
        ],
    }


def _ensure_table(engine) -> None:
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_registry (
                version        SERIAL PRIMARY KEY,
                model_name     TEXT NOT NULL,
                backtest_smape DOUBLE PRECISION,
                training_rows  INTEGER,
                n_skus         INTEGER,
                horizon        INTEGER,
                trained_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                notes          TEXT
            )
        """))


def record_finetune(result: dict, notes: str = "") -> dict:
    """Append a fine-tune run to the logbook. Returns the new version row."""
    engine = get_engine()
    if engine is None:
        return {"version": None, "trained_at": datetime.now().isoformat(), **result}
    _ensure_table(engine)
    from sqlalchemy import text
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO model_registry
                (model_name, backtest_smape, training_rows, n_skus, horizon, notes)
            VALUES (:m, :s, :r, :n, :h, :notes)
            RETURNING version, trained_at
        """), {
            "m": result["model_name"], "s": result["backtest_smape"],
            "r": result["training_rows"], "n": result["n_skus"],
            "h": result["horizon"], "notes": notes,
        }).one()
    return {"version": int(row[0]), "trained_at": row[1].isoformat(), **result}


def get_history(limit: int = 20) -> list[dict]:
    """The fine-tune logbook, newest first."""
    engine = get_engine()
    if engine is None:
        return []
    try:
        _ensure_table(engine)
        from sqlalchemy import text
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT version, model_name, backtest_smape, training_rows,
                       n_skus, horizon, trained_at
                FROM model_registry ORDER BY version DESC LIMIT :l
            """), {"l": limit}).all()
        return [{
            "version": int(r[0]), "model_name": r[1], "backtest_smape": r[2],
            "training_rows": r[3], "n_skus": r[4], "horizon": r[5],
            "trained_at": r[6].isoformat(),
        } for r in rows]
    except Exception as e:
        logger.warning("model_registry history unavailable: %s", e)
        return []


def get_registry() -> dict:
    base = _defaults()
    if _REGISTRY_FILE.exists():
        base.update(json.loads(_REGISTRY_FILE.read_text()))

    history = get_history()
    if history:
        latest = history[0]
        base["last_finetune"] = latest["trained_at"][:10]
        nxt = date.fromisoformat(latest["trained_at"][:10]) + timedelta(days=_FINETUNE_PERIOD_DAYS)
        base["next_finetune"] = nxt.isoformat()
        # Reflect the real latest CatBoost score on the accuracy chart.
        if latest["backtest_smape"] is not None:
            for m in base["models"]:
                if m["name"] == latest["model_name"]:
                    m["backtest_mape"] = latest["backtest_smape"]
    base["history"] = history
    return base
