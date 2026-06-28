from datetime import date
from backend.forecasting.registry import get_registry


def test_registry_keys():
    r = get_registry()
    for k in ("last_finetune", "next_finetune", "weights", "models"):
        assert k in r
    assert set(r["weights"]) == {"chronos", "nhits", "catboost"}
    assert abs(sum(r["weights"].values()) - 1.0) < 1e-6


def test_next_after_last():
    r = get_registry()
    assert date.fromisoformat(r["next_finetune"]) > date.fromisoformat(r["last_finetune"])
