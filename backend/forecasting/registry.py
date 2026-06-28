"""Model registry metadata backing the MLOps dashboard."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from .ensemble import ENSEMBLE_WEIGHTS, _NAMES

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


def get_registry() -> dict:
    if _REGISTRY_FILE.exists():
        data = json.loads(_REGISTRY_FILE.read_text())
        base = _defaults()
        base.update(data)
        return base
    return _defaults()
