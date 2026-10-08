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


def test_every_model_says_whether_its_score_was_measured():
    """The MLOps page charts `backtest_mape`, and three of those values are
    placeholders no backtest in this repository produced. A number a reviewer
    can read off a live page has to carry its provenance, so the flag travels
    with the value and the chart labels anything unmeasured."""
    for m in get_registry()["models"]:
        assert "measured" in m, f"{m['name']} does not say whether it was measured"
        assert isinstance(m["measured"], bool)


def test_the_shipped_defaults_are_not_presented_as_measurements():
    from backend.forecasting.registry import _defaults

    assert [m["measured"] for m in _defaults()["models"]] == [False, False, False]
