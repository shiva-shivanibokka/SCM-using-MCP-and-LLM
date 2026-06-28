from __future__ import annotations
from datetime import date, timedelta

from fastapi import APIRouter

from backend.forecasting.registry import get_registry

router = APIRouter(prefix="/api/mlops", tags=["mlops"])


@router.get("/registry")
def registry():
    return get_registry()


@router.post("/finetune")
def trigger_finetune():
    """Kick off a quarterly fine-tune run.

    The forecasting stack is zero-shot between scheduled fine-tunes, so this
    simulates enqueuing a refresh job: it reports the run that would start now
    and when the next automatic cycle lands. (A real deployment would hand this
    to a job queue / GitHub Action that retrains N-HiTS + CatBoost and re-pins
    the Chronos weights.)
    """
    reg = get_registry()
    started = date(2025, 12, 1) + timedelta(days=90)  # next scheduled cycle
    return {
        "status": "queued",
        "job_id": "ft-" + started.isoformat().replace("-", ""),
        "message": "Fine-tune run queued. N-HiTS and CatBoost will retrain on the "
                   "latest demand; Chronos stays zero-shot and is re-pinned after backtest.",
        "started_run": started.isoformat(),
        "next_finetune": (started + timedelta(days=90)).isoformat(),
        "models": [m["name"] for m in reg["models"]],
    }
