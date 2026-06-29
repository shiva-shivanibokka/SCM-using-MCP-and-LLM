from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.forecasting.registry import get_registry, record_finetune
from backend.forecasting.training import retrain_catboost
from backend.observability import recent_runs

router = APIRouter(prefix="/api/mlops", tags=["mlops"])


@router.get("/registry")
def registry():
    return get_registry()


@router.get("/agent-runs")
def agent_runs(limit: int = 25):
    """Recent agent-run telemetry — the 'receipt' for each assistant turn."""
    return {"runs": recent_runs(limit)}


@router.post("/finetune")
def trigger_finetune():
    """Actually retrain CatBoost and log the run to the model registry.

    Pulls the latest demand for the busiest SKUs, retrains CatBoost, backtests
    it (holding out the most recent horizon and scoring sMAPE), then appends a
    new version row to the model_registry logbook. Chronos stays zero-shot.
    """
    try:
        result = retrain_catboost(n_skus=8, horizon=30)
    except Exception as e:  # surface real training failures to the UI
        raise HTTPException(500, f"retrain failed: {e}")

    rec = record_finetune(result, notes="manual fine-tune via dashboard")
    smape = result["backtest_smape"]
    return {
        "status": "completed",
        "version": rec.get("version"),
        "model": result["model_name"],
        "backtest_smape": smape,
        "n_skus": result["n_skus"],
        "training_rows": result["training_rows"],
        "trained_at": rec.get("trained_at"),
        "message": (
            f"Retrained CatBoost on {result['training_rows']:,} rows across "
            f"{result['n_skus']} SKUs."
            + (f" Backtest sMAPE {smape}%." if smape is not None else "")
        ),
    }
