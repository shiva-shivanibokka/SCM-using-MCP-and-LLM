"""Amazon Chronos-T5-small zero-shot forecaster (CPU, float32)."""
from __future__ import annotations

import threading
import numpy as np

from .contract import validate_forecast

_MODEL_ID = "amazon/chronos-t5-small"
_pipeline = None
_lock = threading.Lock()


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                import torch
                from chronos import ChronosPipeline
                _pipeline = ChronosPipeline.from_pretrained(
                    _MODEL_ID, device_map="cpu", torch_dtype=torch.float32
                )
    return _pipeline


def forecast_chronos(history: list[float], horizon: int = 30) -> dict:
    import torch
    pipe = _get_pipeline()
    context = torch.tensor([float(x) for x in history], dtype=torch.float32)
    forecast = pipe.predict(context, prediction_length=horizon, num_samples=20)
    arr = forecast[0].numpy()  # shape (num_samples, horizon)
    p10 = np.quantile(arr, 0.1, axis=0)
    p50 = np.quantile(arr, 0.5, axis=0)
    p90 = np.quantile(arr, 0.9, axis=0)
    # enforce monotonic ordering defensively
    p10 = np.maximum(p10, 0.0)
    p50 = np.maximum(p50, p10)
    p90 = np.maximum(p90, p50)
    result = {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}
    return validate_forecast(result, horizon)
