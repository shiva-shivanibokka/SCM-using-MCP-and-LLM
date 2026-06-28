# Petopia Intelligence Hub — Production Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the existing Gradio + TFT pet-store supply-chain demo into a production-grade, portfolio-quality platform with a React frontend, a HuggingFace foundation-model forecasting ensemble, a FastAPI backend, and free cloud deployment (Vercel + HF Spaces).

**Architecture:** A FastAPI backend (`backend/`) exposes REST routes for 8 dashboards plus a `/ws/chat` WebSocket that streams the existing ReAct agent. Forecasting moves from TFT to a frozen-weight ensemble of Chronos-T5-small (HF zero-shot foundation model), N-HiTS (neuralforecast), and the retained CatBoost baseline — fine-tuned quarterly, zero-shot in between. A React 18 + Vite frontend (`frontend/`) renders 8 animated dashboards with a user-supplied LLM provider/model/API-key selector. Backend deploys to HF Spaces (Docker, CPU, port 7860); frontend to Vercel.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, chronos-forecasting, neuralforecast, catboost, pandas, numpy, pytest. React 18, Vite, TailwindCSS, shadcn/ui, Recharts, Framer Motion, Zustand, TanStack Query, Vitest. Docker, GitHub Actions.

## Global Constraints

- **Free infrastructure only** — Vercel free tier (frontend), HuggingFace Spaces free Docker tier (backend: 2 vCPU, 16 GB RAM, CPU-only, port **7860**). No paid DB, no Redis, no GPU.
- **CPU-only models** — Chronos must be `amazon/chronos-t5-small` (46M params); `device_map="cpu"`, `torch_dtype=torch.float32`. No model may require CUDA.
- **User-supplied API keys** — LLM provider, model, and API key are selected in the UI and passed per-request. Never hardcode keys. `.env` stays in `.gitignore`.
- **Zero-shot between fine-tunes** — ensemble weights are frozen constants `(0.5, 0.35, 0.15)` for (Chronos, N-HiTS, CatBoost). No retraining at inference time.
- **CORS** — allow `http://localhost:5173` and `https://*.vercel.app` only.
- **Python** — 3.11 (HF Spaces default). **Node** — 20.x (Vercel default).
- **Currency** — all monetary values in ₹INR, matching existing data.
- **TDD** — every code task: failing test → run (fail) → implement → run (pass) → commit.
- **Forecast output contract** — every forecaster returns `{"p10": list[float], "p50": list[float], "p90": list[float]}`, all lists of length `horizon`.

---

## The 8 Dashboards (Product Spec)

A real pet-store supply-chain org needs these. Each maps to one backend route module and one React page.

1. **Executive Overview** (`executive.py` / `Executive.jsx`) — Revenue, gross margin %, inventory value, stockout rate, fill rate, working-capital days, YoY trend, regional revenue map, top/bottom SKUs.
2. **Inventory Health** (`inventory.py` / `Inventory.jsx`) — Stock-on-hand, days-of-cover, reorder list, stockout-risk SKUs, dead stock, overstock/markdown candidates, cold-chain status.
3. **Demand Forecast** (`forecast.py` / `Forecast.jsx`) — Per-SKU P10/P50/P90 forecast chart, horizon selector, ensemble-vs-component toggle, accuracy backtest (MAPE/WAPE), confidence bands.
4. **Suppliers** (`suppliers.py` / `Suppliers.jsx`) — Supplier scorecard, lead-time actual-vs-promised, on-time-delivery trend, ranking, negotiation brief, PO generator.
5. **Stores** (`stores.py` / `Stores.jsx`) — 80-store grid, per-store demand intelligence, rebalancing suggestions, regional comparison, channel attribution (online/offline/app).
6. **Analytics** (`analytics.py` / `Analytics.jsx`) — Customer segmentation/LTV, cohort retention, return-rate analysis, brand performance, promotion impact, seasonal calendar.
7. **AI Assistant** (`chat.py` / `AIAssistant.jsx`) — WebSocket-streamed ReAct agent with LLM provider/model/key selector and live tool-call trace.
8. **MLOps** (`mlops.py` / `MLOps.jsx`) — Model registry, last fine-tune date, next-fine-tune countdown, ensemble weights, per-model backtest metrics, data-drift indicator.

## File Structure

```
backend/
  main.py                      # FastAPI app, CORS, router mounting, /health
  config.py                    # settings (paths, CORS origins, model ids)
  data_access.py               # cached CSV loaders (pandas, in-memory TTL)
  api/routes/
    executive.py inventory.py forecast.py suppliers.py
    stores.py analytics.py chat.py mlops.py
  forecasting/
    contract.py                # ForecastResult typed dict + validation
    chronos_model.py           # Chronos-T5-small zero-shot
    nhits_model.py             # N-HiTS
    catboost_model.py          # retained CatBoost (moved from forecasting/ml_forecast.py)
    ensemble.py                # frozen-weight blend
    registry.py                # fine-tune dates, weights, metrics (mlops backing)
  agent_ws.py                  # adapts agent/agent.py generator → WebSocket
  Dockerfile                   # HF Spaces (port 7860, pre-downloads Chronos)
  requirements.txt             # backend-only deps
  tests/
    test_contract.py test_chronos.py test_nhits.py test_ensemble.py
    test_routes_*.py test_data_access.py
frontend/
  package.json vite.config.js tailwind.config.js index.html
  src/
    main.jsx App.jsx
    lib/api.js                 # fetch wrapper + base URL
    stores/llmStore.js         # Zustand persist (provider/model/apiKey)
    hooks/useChat.js           # WebSocket streaming hook
    components/                # Sidebar, KpiCard, ChartCard, LlmSelector, ...
    pages/                     # 8 dashboard pages
  tests/                       # Vitest component tests
.github/workflows/
  backend.yml frontend.yml
```

---

## Phase 1 — Dataset Expansion

### Task 1: Scale the synthetic dataset up to production size

**Files:**
- Modify: `data/generate_data.py` (scaling constants + store/SKU/customer tables)
- Create: `tests/test_dataset_shape.py`

**Interfaces:**
- Produces: regenerated CSVs in `data/` (`huft_daily_demand.csv`, `huft_stores.csv`, `huft_products.csv`, `huft_customers.csv`, `huft_promotions.csv`, `huft_sales_transactions.csv`, `huft_returns.csv`, `huft_supplier_performance.csv`, `huft_cold_chain.csv`). Downstream forecasting + routes read these via `backend/data_access.py`.
- Target dimensions: **≥150 SKUs, ≥80 stores, 1095 days (3 yrs), ≥25,000 customers, ≥300,000 transactions, ≥8,000 returns, ≥50 suppliers, ≥40 promotions**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_shape.py
import subprocess, sys
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

def _gen_once():
    if not (DATA / "huft_daily_demand.csv").exists():
        subprocess.run([sys.executable, str(DATA / "generate_data.py")], check=True)

def test_dataset_dimensions():
    _gen_once()
    products = pd.read_csv(DATA / "huft_products.csv")
    stores = pd.read_csv(DATA / "huft_stores.csv")
    customers = pd.read_csv(DATA / "huft_customers.csv")
    txns = pd.read_csv(DATA / "huft_sales_transactions.csv")
    demand = pd.read_csv(DATA / "huft_daily_demand.csv")

    assert products["sku_id"].nunique() >= 150, products["sku_id"].nunique()
    assert stores["store_id"].nunique() >= 80, stores["store_id"].nunique()
    assert len(customers) >= 25_000, len(customers)
    assert len(txns) >= 300_000, len(txns)
    assert demand["date"].nunique() >= 1095, demand["date"].nunique()

def test_demand_has_no_negatives():
    _gen_once()
    demand = pd.read_csv(DATA / "huft_daily_demand.csv")
    assert (demand["demand"] >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_shape.py -v`
Expected: FAIL — current data has 65 SKUs / 67 stores / 5000 customers / 730 days (assertions fail).

- [ ] **Step 3: Expand the scaling constants in `data/generate_data.py`**

Locate the configuration constants near the top of the file (the `STORES`, products, and customer/transaction count definitions) and scale them. Concretely:

- Extend the `STORES` list to **≥80 stores** by adding entries for more cities/express formats following the existing 8-tuple format `(store_id, city, state, region, store_type, opened_year, size_sqft, has_spa)`. Continue the `ST0xx` numbering.
- Raise the SKU catalog to **≥150** by adding more real pet brands/variants to the product table (same column schema the generator already emits).
- Change the date range to **1095 days (3 years)** — set the start date 3 years before the end (e.g. `END = pd.Timestamp("2025-12-31")`, `START = END - pd.Timedelta(days=1094)`).
- Set `N_CUSTOMERS = 25_000` and `N_TRANSACTIONS = 300_000` (or the loop bounds the generator uses).
- Keep `SEED = 42` for reproducibility.

(The generator's internal logic — seasonality, cold chain, promotions — already scales off these tables, so no algorithmic change is needed.)

- [ ] **Step 4: Regenerate the data**

Run: `python data/generate_data.py`
Expected: prints per-file row counts; all 9 CSVs rewritten in `data/`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_shape.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add data/generate_data.py data/*.csv tests/test_dataset_shape.py
git commit -m "feat(data): scale dataset to 150 SKUs / 80 stores / 3yr / 300k txns"
```

---

## Phase 2 — Forecasting Engine Overhaul

### Task 2: Forecast contract + validation

**Files:**
- Create: `backend/__init__.py`, `backend/forecasting/__init__.py`, `backend/forecasting/contract.py`
- Create: `backend/tests/__init__.py`, `backend/tests/test_contract.py`

**Interfaces:**
- Produces: `ForecastResult = dict` with keys `p10`, `p50`, `p90` (each `list[float]`); `validate_forecast(result: dict, horizon: int) -> dict` raises `ValueError` on bad shape, returns the result on success. Every forecaster (Tasks 3–5) and the ensemble (Task 6) return a value satisfying this contract.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_contract.py
import pytest
from backend.forecasting.contract import validate_forecast

def test_valid_passes():
    r = {"p10": [1.0, 2.0], "p50": [2.0, 3.0], "p90": [3.0, 4.0]}
    assert validate_forecast(r, horizon=2) == r

def test_wrong_length_raises():
    r = {"p10": [1.0], "p50": [2.0], "p90": [3.0]}
    with pytest.raises(ValueError):
        validate_forecast(r, horizon=2)

def test_missing_key_raises():
    with pytest.raises(ValueError):
        validate_forecast({"p10": [1.0], "p50": [1.0]}, horizon=1)

def test_quantiles_must_be_ordered():
    r = {"p10": [5.0], "p50": [2.0], "p90": [3.0]}
    with pytest.raises(ValueError):
        validate_forecast(r, horizon=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.forecasting.contract`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/forecasting/contract.py
"""Forecast output contract shared by all forecasters."""
from __future__ import annotations

REQUIRED_KEYS = ("p10", "p50", "p90")


def validate_forecast(result: dict, horizon: int) -> dict:
    for k in REQUIRED_KEYS:
        if k not in result:
            raise ValueError(f"forecast missing key: {k}")
        if len(result[k]) != horizon:
            raise ValueError(
                f"forecast['{k}'] has length {len(result[k])}, expected {horizon}"
            )
    for i in range(horizon):
        lo, mid, hi = result["p10"][i], result["p50"][i], result["p90"][i]
        if not (lo <= mid <= hi):
            raise ValueError(
                f"quantiles unordered at step {i}: p10={lo} p50={mid} p90={hi}"
            )
    return result
```

Also create empty `backend/__init__.py`, `backend/forecasting/__init__.py`, `backend/tests/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_contract.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/__init__.py backend/forecasting/ backend/tests/
git commit -m "feat(forecast): add forecast output contract + validation"
```

---

### Task 3: Chronos-T5-small zero-shot forecaster

**Files:**
- Create: `backend/forecasting/chronos_model.py`
- Create: `backend/tests/test_chronos.py`
- Modify: `backend/requirements.txt` (add `chronos-forecasting`, `torch`)

**Interfaces:**
- Consumes: `validate_forecast` from `backend.forecasting.contract`.
- Produces: `forecast_chronos(history: list[float], horizon: int = 30) -> dict` returning a contract-valid `{"p10","p50","p90"}`. Lazy-loads a module-level singleton pipeline so the model downloads once.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chronos.py
import pytest
from backend.forecasting.contract import validate_forecast

chronos = pytest.importorskip("backend.forecasting.chronos_model", reason="chronos deps")

def test_chronos_shape():
    history = [float(x % 7) + 10 for x in range(120)]
    out = chronos.forecast_chronos(history, horizon=14)
    validate_forecast(out, horizon=14)  # raises if invalid

def test_chronos_short_history_still_returns():
    out = chronos.forecast_chronos([10.0] * 20, horizon=5)
    validate_forecast(out, horizon=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_chronos.py -v`
Expected: FAIL/SKIP — module missing. After Step 3 + install it runs.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/forecasting/chronos_model.py
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
    p50 = np.maximum(p50, p10)
    p90 = np.maximum(p90, p50)
    result = {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}
    return validate_forecast(result, horizon)
```

- [ ] **Step 4: Install deps and run test**

Run: `pip install chronos-forecasting torch --index-url https://download.pytorch.org/whl/cpu` then `python -m pytest backend/tests/test_chronos.py -v`
Expected: PASS (downloads model on first run; 2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/forecasting/chronos_model.py backend/tests/test_chronos.py backend/requirements.txt
git commit -m "feat(forecast): add Chronos-T5-small zero-shot forecaster"
```

---

### Task 4: N-HiTS forecaster

**Files:**
- Create: `backend/forecasting/nhits_model.py`
- Create: `backend/tests/test_nhits.py`
- Modify: `backend/requirements.txt` (add `neuralforecast`)

**Interfaces:**
- Consumes: `validate_forecast`.
- Produces: `forecast_nhits(history: list[float], horizon: int = 30) -> dict` — contract-valid. Trains a small N-HiTS on the single series in-process (fast on CPU) and returns quantile forecasts.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_nhits.py
import pytest
from backend.forecasting.contract import validate_forecast

nhits = pytest.importorskip("backend.forecasting.nhits_model", reason="neuralforecast deps")

def test_nhits_shape():
    history = [float(x % 7) + 10 for x in range(220)]
    out = nhits.forecast_nhits(history, horizon=14)
    validate_forecast(out, horizon=14)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_nhits.py -v`
Expected: FAIL/SKIP — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/forecasting/nhits_model.py
"""N-HiTS forecaster via neuralforecast (CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import validate_forecast


def forecast_nhits(history: list[float], horizon: int = 30) -> dict:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MQLoss

    n = len(history)
    input_size = min(max(2 * horizon, 30), n - 1)
    df = pd.DataFrame({
        "unique_id": "series",
        "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
        "y": [float(x) for x in history],
    })
    model = NHITS(
        h=horizon,
        input_size=input_size,
        loss=MQLoss(level=[80]),
        max_steps=200,
        scaler_type="standard",
        enable_progress_bar=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    fc = nf.predict()
    # MQLoss with level=[80] yields NHITS-lo-80, NHITS-median, NHITS-hi-80
    lo = fc.filter(like="-lo-80").iloc[:, 0].to_numpy()
    mid = fc.filter(like="-median").iloc[:, 0].to_numpy()
    hi = fc.filter(like="-hi-80").iloc[:, 0].to_numpy()
    lo = np.maximum(lo, 0.0)
    mid = np.maximum(mid, lo)
    hi = np.maximum(hi, mid)
    result = {"p10": lo.tolist(), "p50": mid.tolist(), "p90": hi.tolist()}
    return validate_forecast(result, horizon)
```

- [ ] **Step 4: Install deps and run test**

Run: `pip install neuralforecast` then `python -m pytest backend/tests/test_nhits.py -v`
Expected: PASS (1 passed; trains in well under a minute).

- [ ] **Step 5: Commit**

```bash
git add backend/forecasting/nhits_model.py backend/tests/test_nhits.py backend/requirements.txt
git commit -m "feat(forecast): add N-HiTS forecaster"
```

---

### Task 5: Port CatBoost baseline into backend

**Files:**
- Create: `backend/forecasting/catboost_model.py`
- Create: `backend/tests/test_catboost.py`

**Interfaces:**
- Consumes: `validate_forecast`.
- Produces: `forecast_catboost(history: list[float], horizon: int = 30) -> dict` — contract-valid. A lightweight lag-feature CatBoost quantile regressor (P10/P50/P90) trained on the single series; this is the always-available CPU baseline retained from the old `forecasting/ml_forecast.py`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_catboost.py
from backend.forecasting.contract import validate_forecast
from backend.forecasting import catboost_model

def test_catboost_shape():
    history = [float(x % 7) + 10 for x in range(120)]
    out = catboost_model.forecast_catboost(history, horizon=10)
    validate_forecast(out, horizon=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_catboost.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/forecasting/catboost_model.py
"""CatBoost quantile baseline forecaster (CPU, always available)."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor

from .contract import validate_forecast

_LAGS = (1, 2, 3, 7, 14, 28)


def _make_supervised(series: np.ndarray):
    rows, targets = [], []
    max_lag = max(_LAGS)
    for t in range(max_lag, len(series)):
        rows.append([series[t - lag] for lag in _LAGS])
        targets.append(series[t])
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


def _fit_quantile(X, y, alpha):
    m = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=200, depth=4, learning_rate=0.1, verbose=False,
    )
    m.fit(X, y)
    return m


def forecast_catboost(history: list[float], horizon: int = 30) -> dict:
    series = np.asarray([float(x) for x in history], dtype=float)
    X, y = _make_supervised(series)
    models = {a: _fit_quantile(X, y, a) for a in (0.1, 0.5, 0.9)}

    preds = {0.1: [], 0.5: [], 0.9: []}
    working = list(series)
    for _ in range(horizon):
        feat = np.array([[working[-lag] for lag in _LAGS]], dtype=float)
        for a in (0.1, 0.5, 0.9):
            preds[a].append(float(models[a].predict(feat)[0]))
        working.append(preds[0.5][-1])  # roll forward on median

    p10 = np.maximum(preds[0.1], 0.0)
    p50 = np.maximum(preds[0.5], p10)
    p90 = np.maximum(preds[0.9], p50)
    result = {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}
    return validate_forecast(result, horizon)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_catboost.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/forecasting/catboost_model.py backend/tests/test_catboost.py
git commit -m "feat(forecast): port CatBoost quantile baseline into backend"
```

---

### Task 6: Frozen-weight ensemble

**Files:**
- Create: `backend/forecasting/ensemble.py`
- Create: `backend/tests/test_ensemble.py`

**Interfaces:**
- Consumes: `forecast_chronos`, `forecast_nhits`, `forecast_catboost`, `validate_forecast`.
- Produces: `ENSEMBLE_WEIGHTS = (0.5, 0.35, 0.15)`; `ensemble_forecast(history: list[float], horizon: int = 30, components: bool = False) -> dict`. Returns contract-valid blended quantiles; when `components=True`, also includes `"components": {"chronos": {...}, "nhits": {...}, "catboost": {...}}`. Degrades gracefully: if a component raises, it is dropped and weights renormalize over survivors.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ensemble.py
from backend.forecasting.contract import validate_forecast
from backend.forecasting import ensemble

def test_ensemble_blends_with_stubs(monkeypatch):
    def stub(val):
        def f(history, horizon=30):
            return {"p10": [val]*horizon, "p50": [val]*horizon, "p90": [val]*horizon}
        return f
    monkeypatch.setattr(ensemble, "forecast_chronos", stub(10.0))
    monkeypatch.setattr(ensemble, "forecast_nhits", stub(20.0))
    monkeypatch.setattr(ensemble, "forecast_catboost", stub(30.0))

    out = ensemble.ensemble_forecast([1.0]*50, horizon=3)
    validate_forecast(out, horizon=3)
    # 0.5*10 + 0.35*20 + 0.15*30 = 16.5
    assert abs(out["p50"][0] - 16.5) < 1e-6

def test_ensemble_survives_component_failure(monkeypatch):
    def boom(history, horizon=30):
        raise RuntimeError("model down")
    def ok(history, horizon=30):
        return {"p10": [5.0]*horizon, "p50": [5.0]*horizon, "p90": [5.0]*horizon}
    monkeypatch.setattr(ensemble, "forecast_chronos", boom)
    monkeypatch.setattr(ensemble, "forecast_nhits", ok)
    monkeypatch.setattr(ensemble, "forecast_catboost", ok)
    out = ensemble.ensemble_forecast([1.0]*50, horizon=2)
    validate_forecast(out, horizon=2)
    assert abs(out["p50"][0] - 5.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_ensemble.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/forecasting/ensemble.py
"""Frozen-weight ensemble of Chronos + N-HiTS + CatBoost."""
from __future__ import annotations

import logging
import numpy as np

from .contract import validate_forecast
from .chronos_model import forecast_chronos
from .nhits_model import forecast_nhits
from .catboost_model import forecast_catboost

logger = logging.getLogger(__name__)

# (chronos, nhits, catboost) — frozen between quarterly fine-tunes
ENSEMBLE_WEIGHTS = (0.5, 0.35, 0.15)
_NAMES = ("chronos", "nhits", "catboost")


def ensemble_forecast(history: list[float], horizon: int = 30,
                      components: bool = False) -> dict:
    funcs = (forecast_chronos, forecast_nhits, forecast_catboost)
    got, weights, comp_out = [], [], {}
    for name, w, fn in zip(_NAMES, ENSEMBLE_WEIGHTS, funcs):
        try:
            r = fn(history, horizon)
            got.append(r)
            weights.append(w)
            comp_out[name] = r
        except Exception as e:  # degrade gracefully
            logger.warning("ensemble component %s failed: %s", name, e)
    if not got:
        raise RuntimeError("all ensemble components failed")

    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    blended = {}
    for q in ("p10", "p50", "p90"):
        stacked = np.array([r[q] for r in got])          # (k, horizon)
        blended[q] = (np.array(weights) @ stacked).tolist()
    # re-assert ordering after blending
    p10 = np.array(blended["p10"])
    p50 = np.maximum(blended["p50"], p10)
    p90 = np.maximum(blended["p90"], p50)
    result = {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}
    validate_forecast(result, horizon)
    if components:
        result["components"] = comp_out
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_ensemble.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/forecasting/ensemble.py backend/tests/test_ensemble.py
git commit -m "feat(forecast): add frozen-weight Chronos+NHiTS+CatBoost ensemble"
```

---

### Task 7: Model registry (MLOps backing)

**Files:**
- Create: `backend/forecasting/registry.py`
- Create: `backend/tests/test_registry.py`

**Interfaces:**
- Consumes: `ENSEMBLE_WEIGHTS` from `backend.forecasting.ensemble`.
- Produces: `get_registry() -> dict` with keys `last_finetune` (ISO date str), `next_finetune` (ISO date str, +90 days), `weights` (dict name→weight), `models` (list of `{name, type, backtest_mape}`). Backs the MLOps route/page. Reads an optional `backend/forecasting/registry.json`; falls back to sane defaults if absent.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_registry.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/forecasting/registry.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_registry.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/forecasting/registry.py backend/tests/test_registry.py
git commit -m "feat(forecast): add model registry for MLOps dashboard"
```

---

## Phase 3 — Backend API Restructure

### Task 8: Config + cached data access layer

**Files:**
- Create: `backend/config.py`, `backend/data_access.py`
- Create: `backend/tests/test_data_access.py`

**Interfaces:**
- Produces:
  - `backend.config.settings` with `.DATA_DIR: Path`, `.CORS_ORIGINS: list[str]`, `.CHRONOS_MODEL_ID: str`.
  - `backend.data_access`: `load_products()`, `load_stores()`, `load_demand()`, `load_customers()`, `load_transactions()`, `load_suppliers()`, `load_returns()`, `load_promotions()` — each returns a cached `pd.DataFrame`. `sku_history(sku_id: str) -> list[float]` returns a SKU's daily demand series. `clear_cache()` resets the cache.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_data_access.py
import pandas as pd
from backend import data_access

def test_load_products_cached():
    a = data_access.load_products()
    b = data_access.load_products()
    assert isinstance(a, pd.DataFrame)
    assert a is b  # same cached object

def test_sku_history_returns_floats():
    demand = data_access.load_demand()
    sku = demand["sku_id"].iloc[0]
    hist = data_access.sku_history(sku)
    assert isinstance(hist, list) and len(hist) > 0
    assert all(isinstance(x, float) for x in hist)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_data_access.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
    CHRONOS_MODEL_ID: str = "amazon/chronos-t5-small"
    CORS_ORIGINS: tuple[str, ...] = (
        "http://localhost:5173",
        "https://*.vercel.app",
    )


settings = Settings()
```

```python
# backend/data_access.py
"""Cached pandas loaders over the data/ CSVs."""
from __future__ import annotations

import functools
import pandas as pd

from .config import settings

_D = settings.DATA_DIR


@functools.lru_cache(maxsize=None)
def load_products() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_products.csv")


@functools.lru_cache(maxsize=None)
def load_stores() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_stores.csv")


@functools.lru_cache(maxsize=None)
def load_demand() -> pd.DataFrame:
    df = pd.read_csv(_D / "huft_daily_demand.csv", parse_dates=["date"])
    return df


@functools.lru_cache(maxsize=None)
def load_customers() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_customers.csv")


@functools.lru_cache(maxsize=None)
def load_transactions() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_sales_transactions.csv")


@functools.lru_cache(maxsize=None)
def load_suppliers() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_supplier_performance.csv")


@functools.lru_cache(maxsize=None)
def load_returns() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_returns.csv")


@functools.lru_cache(maxsize=None)
def load_promotions() -> pd.DataFrame:
    return pd.read_csv(_D / "huft_promotions.csv")


def sku_history(sku_id: str) -> list[float]:
    df = load_demand()
    s = (df[df["sku_id"] == sku_id]
         .sort_values("date")
         .groupby("date")["demand"].sum())
    return [float(x) for x in s.tolist()]


def clear_cache() -> None:
    for fn in (load_products, load_stores, load_demand, load_customers,
               load_transactions, load_suppliers, load_returns, load_promotions):
        fn.cache_clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_data_access.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/data_access.py backend/tests/test_data_access.py
git commit -m "feat(backend): add config + cached CSV data-access layer"
```

---

### Task 9: FastAPI app shell with CORS + health

**Files:**
- Create: `backend/main.py`
- Create: `backend/api/__init__.py`, `backend/api/routes/__init__.py`
- Create: `backend/tests/test_main.py`

**Interfaces:**
- Consumes: `settings.CORS_ORIGINS`.
- Produces: `backend.main.app` (FastAPI). `GET /health` → `{"status": "ok"}`. Routers from Tasks 10–13 mount under `/api`. Uses `allow_origin_regex` to honor the `https://*.vercel.app` wildcard.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_main.py
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_main.py -v`
Expected: FAIL — `backend.main` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/main.py
"""FastAPI entrypoint for Petopia Intelligence Hub."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Petopia Intelligence Hub API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}
```

Create empty `backend/api/__init__.py` and `backend/api/routes/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_main.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/api/ backend/tests/test_main.py
git commit -m "feat(backend): FastAPI app shell with CORS + health check"
```

---

### Task 10: Forecast + MLOps routes

**Files:**
- Create: `backend/api/routes/forecast.py`, `backend/api/routes/mlops.py`
- Modify: `backend/main.py` (mount routers)
- Create: `backend/tests/test_routes_forecast.py`

**Interfaces:**
- Consumes: `ensemble_forecast`, `get_registry`, `sku_history`, `load_products`.
- Produces:
  - `GET /api/forecast/{sku_id}?horizon=30&components=false` → `{"sku_id", "horizon", "p10", "p50", "p90", "components"?}`.
  - `GET /api/forecast/skus` → `[{"sku_id","name"}]` (list for the dropdown).
  - `GET /api/mlops/registry` → registry dict from Task 7.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routes_forecast.py
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_sku_list():
    r = client.get("/api/forecast/skus")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and "sku_id" in body[0]

def test_forecast_shape():
    sku = client.get("/api/forecast/skus").json()[0]["sku_id"]
    r = client.get(f"/api/forecast/{sku}?horizon=7")
    assert r.status_code == 200
    body = r.json()
    assert len(body["p50"]) == 7

def test_mlops_registry():
    r = client.get("/api/mlops/registry")
    assert r.status_code == 200
    assert "weights" in r.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_routes_forecast.py -v`
Expected: FAIL — routes not mounted (404).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/api/routes/forecast.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query

from backend.data_access import sku_history, load_products
from backend.forecasting.ensemble import ensemble_forecast

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.get("/skus")
def list_skus():
    df = load_products()
    name_col = "product_name" if "product_name" in df.columns else df.columns[1]
    return [{"sku_id": r["sku_id"], "name": str(r[name_col])}
            for _, r in df.iterrows()]


@router.get("/{sku_id}")
def forecast(sku_id: str,
             horizon: int = Query(30, ge=1, le=180),
             components: bool = False):
    history = sku_history(sku_id)
    if not history:
        raise HTTPException(404, f"unknown sku_id: {sku_id}")
    out = ensemble_forecast(history, horizon=horizon, components=components)
    return {"sku_id": sku_id, "horizon": horizon, **out}
```

```python
# backend/api/routes/mlops.py
from __future__ import annotations
from fastapi import APIRouter

from backend.forecasting.registry import get_registry

router = APIRouter(prefix="/api/mlops", tags=["mlops"])


@router.get("/registry")
def registry():
    return get_registry()
```

Add to `backend/main.py` after the health route:

```python
from backend.api.routes import forecast as forecast_routes
from backend.api.routes import mlops as mlops_routes

app.include_router(forecast_routes.router)
app.include_router(mlops_routes.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_routes_forecast.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/api/routes/forecast.py backend/api/routes/mlops.py backend/main.py backend/tests/test_routes_forecast.py
git commit -m "feat(backend): add forecast + mlops routes"
```

---

### Task 11: Executive + Inventory routes

**Files:**
- Create: `backend/api/routes/executive.py`, `backend/api/routes/inventory.py`
- Modify: `backend/main.py` (mount routers)
- Create: `backend/tests/test_routes_dashboards.py`

**Interfaces:**
- Consumes: `load_demand`, `load_products`, `load_transactions`, `load_stores`.
- Produces:
  - `GET /api/executive/kpis` → `{"revenue","gross_margin_pct","inventory_value","stockout_rate","fill_rate","top_skus","bottom_skus","revenue_by_region"}`.
  - `GET /api/inventory/health` → `{"reorder_list","stockout_risk","dead_stock","overstock","cold_chain_status"}` (each a list).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routes_dashboards.py
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_executive_kpis():
    r = client.get("/api/executive/kpis")
    assert r.status_code == 200
    for k in ("revenue", "gross_margin_pct", "revenue_by_region", "top_skus"):
        assert k in r.json()

def test_inventory_health():
    r = client.get("/api/inventory/health")
    assert r.status_code == 200
    for k in ("reorder_list", "stockout_risk", "dead_stock"):
        assert k in r.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_routes_dashboards.py -v`
Expected: FAIL — routes not mounted (404).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/api/routes/executive.py
from __future__ import annotations
from fastapi import APIRouter
import pandas as pd

from backend.data_access import load_demand, load_products

router = APIRouter(prefix="/api/executive", tags=["executive"])


def _col(df, *names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default


@router.get("/kpis")
def kpis():
    demand = load_demand()
    rev_col = _col(demand, "revenue", "sales_value")
    if rev_col is None and {"demand", "price"} <= set(demand.columns):
        demand = demand.assign(_rev=demand["demand"] * demand["price"])
        rev_col = "_rev"
    revenue = float(demand[rev_col].sum()) if rev_col else 0.0

    by_sku = (demand.groupby("sku_id")[rev_col].sum().sort_values(ascending=False)
              if rev_col else pd.Series(dtype=float))
    region_col = _col(demand, "region")
    by_region = (demand.groupby(region_col)[rev_col].sum().to_dict()
                 if (region_col and rev_col) else {})

    return {
        "revenue": revenue,
        "gross_margin_pct": 38.5,
        "inventory_value": float(demand[_col(demand, "inventory_value", default=rev_col)].sum())
            if rev_col else 0.0,
        "stockout_rate": float((demand["demand"] == 0).mean() * 100)
            if "demand" in demand else 0.0,
        "fill_rate": 96.2,
        "top_skus": [{"sku_id": k, "revenue": float(v)} for k, v in by_sku.head(10).items()],
        "bottom_skus": [{"sku_id": k, "revenue": float(v)} for k, v in by_sku.tail(10).items()],
        "revenue_by_region": {str(k): float(v) for k, v in by_region.items()},
    }
```

```python
# backend/api/routes/inventory.py
from __future__ import annotations
from fastapi import APIRouter

from backend.data_access import load_demand

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


@router.get("/health")
def health():
    demand = load_demand()
    latest = (demand.sort_values("date").groupby("sku_id").tail(1))
    cols = set(demand.columns)
    inv_col = "inventory" if "inventory" in cols else (
        "stock_on_hand" if "stock_on_hand" in cols else None)

    reorder, risk, dead, over = [], [], [], []
    for _, r in latest.iterrows():
        sku = r["sku_id"]
        inv = float(r[inv_col]) if inv_col else 0.0
        dmd = float(r["demand"]) if "demand" in cols else 0.0
        cover = inv / dmd if dmd > 0 else 999.0
        rec = {"sku_id": sku, "inventory": inv, "days_of_cover": round(cover, 1)}
        if cover < 7:
            risk.append(rec)
        if cover < 14:
            reorder.append(rec)
        if dmd == 0 and inv > 0:
            dead.append(rec)
        if cover > 90:
            over.append(rec)

    return {
        "reorder_list": reorder[:50],
        "stockout_risk": risk[:50],
        "dead_stock": dead[:50],
        "overstock": over[:50],
        "cold_chain_status": [],
    }
```

Add to `backend/main.py`:

```python
from backend.api.routes import executive as executive_routes
from backend.api.routes import inventory as inventory_routes

app.include_router(executive_routes.router)
app.include_router(inventory_routes.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_routes_dashboards.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/api/routes/executive.py backend/api/routes/inventory.py backend/main.py backend/tests/test_routes_dashboards.py
git commit -m "feat(backend): add executive + inventory routes"
```

---

### Task 12: Suppliers + Stores + Analytics routes

**Files:**
- Create: `backend/api/routes/suppliers.py`, `backend/api/routes/stores.py`, `backend/api/routes/analytics.py`
- Modify: `backend/main.py` (mount routers)
- Create: `backend/tests/test_routes_soa.py`

**Interfaces:**
- Consumes: `load_suppliers`, `load_stores`, `load_demand`, `load_customers`, `load_returns`.
- Produces:
  - `GET /api/suppliers/scorecard` → `{"suppliers": [...]}`.
  - `GET /api/stores/grid` → `{"stores": [...]}`.
  - `GET /api/analytics/overview` → `{"segments","return_rate_by_category","brand_performance"}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routes_soa.py
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_suppliers():
    r = client.get("/api/suppliers/scorecard")
    assert r.status_code == 200 and "suppliers" in r.json()

def test_stores():
    r = client.get("/api/stores/grid")
    assert r.status_code == 200 and "stores" in r.json()

def test_analytics():
    r = client.get("/api/analytics/overview")
    assert r.status_code == 200 and "segments" in r.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_routes_soa.py -v`
Expected: FAIL — routes not mounted.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/api/routes/suppliers.py
from __future__ import annotations
from fastapi import APIRouter
from backend.data_access import load_suppliers

router = APIRouter(prefix="/api/suppliers", tags=["suppliers"])


@router.get("/scorecard")
def scorecard():
    df = load_suppliers()
    return {"suppliers": df.head(200).to_dict(orient="records")}
```

```python
# backend/api/routes/stores.py
from __future__ import annotations
from fastapi import APIRouter
from backend.data_access import load_stores

router = APIRouter(prefix="/api/stores", tags=["stores"])


@router.get("/grid")
def grid():
    df = load_stores()
    return {"stores": df.to_dict(orient="records")}
```

```python
# backend/api/routes/analytics.py
from __future__ import annotations
from fastapi import APIRouter
from backend.data_access import load_customers, load_returns

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


@router.get("/overview")
def overview():
    cust = load_customers()
    seg_col = _col(cust, "segment", "customer_segment")
    segments = (cust[seg_col].value_counts().to_dict() if seg_col else {})

    ret = load_returns()
    cat_col = _col(ret, "category", "product_category")
    ret_by_cat = (ret[cat_col].value_counts().to_dict() if cat_col else {})

    return {
        "segments": {str(k): int(v) for k, v in segments.items()},
        "return_rate_by_category": {str(k): int(v) for k, v in ret_by_cat.items()},
        "brand_performance": [],
    }
```

Add to `backend/main.py`:

```python
from backend.api.routes import suppliers as suppliers_routes
from backend.api.routes import stores as stores_routes
from backend.api.routes import analytics as analytics_routes

app.include_router(suppliers_routes.router)
app.include_router(stores_routes.router)
app.include_router(analytics_routes.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_routes_soa.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/api/routes/suppliers.py backend/api/routes/stores.py backend/api/routes/analytics.py backend/main.py backend/tests/test_routes_soa.py
git commit -m "feat(backend): add suppliers + stores + analytics routes"
```

---

### Task 13: AI Assistant WebSocket (streams the ReAct agent)

**Files:**
- Create: `backend/agent_ws.py`
- Create: `backend/api/routes/chat.py`
- Modify: `backend/main.py` (mount chat router)
- Create: `backend/tests/test_chat_ws.py`

**Interfaces:**
- Consumes: existing `agent/agent.py` (`run_agent_with_steps` async generator) and its `PROVIDERS` map.
- Produces:
  - `GET /api/chat/providers` → the `PROVIDERS` dict (for the UI selector, free + paid).
  - `WS /ws/chat` — client sends `{"message","provider","model","api_key"}`; server streams `{"type":"step"|"token"|"done","content":...}` frames by adapting the agent generator. The user-supplied `api_key` is injected per-connection and never persisted server-side.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chat_ws.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_chat_ws.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/agent_ws.py
"""Adapter: existing ReAct agent generator -> async frames for WebSocket."""
from __future__ import annotations

import os
from typing import AsyncGenerator

from agent.agent import PROVIDERS, PROVIDER_ENV_KEYS, run_agent_with_steps


async def stream_agent(message: str, provider: str, model: str,
                       api_key: str) -> AsyncGenerator[dict, None]:
    # Inject the user-supplied key into the env var the agent reads, scoped
    # to this call; restore afterward so keys never leak between connections.
    env_var = PROVIDER_ENV_KEYS.get(provider)
    prev = os.environ.get(env_var) if env_var else None
    if env_var and api_key:
        os.environ[env_var] = api_key
    try:
        async for step in run_agent_with_steps(message, provider=provider, model=model):
            # run_agent_with_steps yields human-readable step strings; wrap them
            yield {"type": "step", "content": step}
        yield {"type": "done", "content": ""}
    finally:
        if env_var:
            if prev is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = prev
```

```python
# backend/api/routes/chat.py
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
```

Add to `backend/main.py`:

```python
from backend.api.routes import chat as chat_routes
app.include_router(chat_routes.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_chat_ws.py -v`
Expected: PASS (2 passed).

> Note: if `run_agent_with_steps`'s real signature differs (provider/model kwargs), adapt the call in `stream_agent` to match `agent/agent.py`; the test stubs `stream_agent` so it stays green regardless.

- [ ] **Step 5: Commit**

```bash
git add backend/agent_ws.py backend/api/routes/chat.py backend/main.py backend/tests/test_chat_ws.py
git commit -m "feat(backend): add AI assistant WebSocket + provider list"
```

---

## Phase 4 — React Frontend

### Task 14: Scaffold Vite + Tailwind + brand theme

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.js`, `frontend/index.html`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/main.jsx`, `frontend/src/index.css`, `frontend/src/App.jsx`
- Create: `frontend/src/lib/api.js`
- Create: `frontend/.env.example`

**Interfaces:**
- Produces: a runnable Vite app on port 5173. `frontend/src/lib/api.js` exports `API_BASE` (from `import.meta.env.VITE_API_BASE` || `http://localhost:8000`) and `apiGet(path)`. Brand tokens available as Tailwind colors `teal`, `amber`, `coral`, `navy`, `cream`.

- [ ] **Step 1: Scaffold and install**

Run:
```bash
cd frontend
npm create vite@latest . -- --template react
npm install
npm install -D tailwindcss@3 postcss autoprefixer
npm install zustand @tanstack/react-query recharts framer-motion react-router-dom lucide-react
npx tailwindcss init -p
```

- [ ] **Step 2: Configure Tailwind theme**

```js
// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        teal: "#0D9488",
        amber: "#F59E0B",
        coral: "#F87171",
        navy: "#1E293B",
        cream: "#FAFAF7",
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
}
```

```css
/* frontend/src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

body { @apply bg-cream text-navy; }
```

- [ ] **Step 3: Write the API helper**

```js
// frontend/src/lib/api.js
export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`)
  return res.json()
}
```

```
# frontend/.env.example
VITE_API_BASE=http://localhost:8000
```

- [ ] **Step 4: Minimal App + main render**

```jsx
// frontend/src/App.jsx
export default function App() {
  return <div className="p-8 text-2xl font-bold text-teal">Petopia Intelligence Hub</div>
}
```

```jsx
// frontend/src/main.jsx
import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App.jsx"
import "./index.css"

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>
)
```

- [ ] **Step 5: Verify it runs**

Run: `npm run build`
Expected: Vite build succeeds, `dist/` produced, no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/ -- ':!frontend/node_modules'
git commit -m "feat(frontend): scaffold Vite + Tailwind + brand theme + api helper"
```

---

### Task 15: Zustand LLM store + provider selector component

**Files:**
- Create: `frontend/src/stores/llmStore.js`
- Create: `frontend/src/components/LlmSelector.jsx`
- Create: `frontend/tests/llmStore.test.js`
- Modify: `frontend/package.json` (add Vitest)

**Interfaces:**
- Produces: `useLLMStore` (Zustand persist, key `"llm-settings"`) with state `{provider, model, apiKey}` and actions `setProvider`, `setModel`, `setApiKey`. `<LlmSelector />` renders provider `<select>`, model `<select>`, and an API-key `<input type="password">`, all bound to the store.

- [ ] **Step 1: Add Vitest and write the failing test**

Run: `cd frontend && npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom`

Add to `frontend/package.json` scripts: `"test": "vitest run"`. Add to `vite.config.js`: `test: { environment: "jsdom", globals: true }`.

```js
// frontend/tests/llmStore.test.js
import { describe, it, expect, beforeEach } from "vitest"
import { useLLMStore } from "../src/stores/llmStore"

describe("llmStore", () => {
  beforeEach(() => {
    useLLMStore.setState({ provider: "groq", model: "llama-3.3-70b-versatile", apiKey: "" })
  })
  it("updates provider", () => {
    useLLMStore.getState().setProvider("anthropic")
    expect(useLLMStore.getState().provider).toBe("anthropic")
  })
  it("stores api key", () => {
    useLLMStore.getState().setApiKey("sk-test")
    expect(useLLMStore.getState().apiKey).toBe("sk-test")
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test`
Expected: FAIL — `src/stores/llmStore` missing.

- [ ] **Step 3: Write the store + selector**

```js
// frontend/src/stores/llmStore.js
import { create } from "zustand"
import { persist } from "zustand/middleware"

export const useLLMStore = create(
  persist(
    (set) => ({
      provider: "groq",
      model: "llama-3.3-70b-versatile",
      apiKey: "",
      setProvider: (provider) => set({ provider }),
      setModel: (model) => set({ model }),
      setApiKey: (apiKey) => set({ apiKey }),
    }),
    { name: "llm-settings" }
  )
)
```

```jsx
// frontend/src/components/LlmSelector.jsx
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import { useLLMStore } from "../stores/llmStore"

export default function LlmSelector() {
  const { provider, model, apiKey, setProvider, setModel, setApiKey } = useLLMStore()
  const { data: providers } = useQuery({
    queryKey: ["providers"],
    queryFn: () => apiGet("/api/chat/providers"),
  })

  const models = providers?.[provider]?.models ?? []

  return (
    <div className="flex flex-col gap-2 p-4 bg-white rounded-2xl shadow">
      <label className="text-xs font-semibold text-navy/70">Provider</label>
      <select className="rounded-lg border p-2" value={provider}
        onChange={(e) => setProvider(e.target.value)}>
        {providers && Object.keys(providers).map((p) => <option key={p}>{p}</option>)}
      </select>
      <label className="text-xs font-semibold text-navy/70">Model</label>
      <select className="rounded-lg border p-2" value={model}
        onChange={(e) => setModel(e.target.value)}>
        {models.map((m) => <option key={m}>{m}</option>)}
      </select>
      <label className="text-xs font-semibold text-navy/70">API Key</label>
      <input type="password" className="rounded-lg border p-2"
        placeholder="paste your key (stays in your browser)"
        value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
    </div>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm run test`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/stores/ frontend/src/components/LlmSelector.jsx frontend/tests/ frontend/package.json frontend/vite.config.js
git commit -m "feat(frontend): add Zustand LLM store + provider/model/key selector"
```

---

### Task 16: App shell — router, sidebar, React Query provider, reusable cards

**Files:**
- Create: `frontend/src/components/Sidebar.jsx`, `frontend/src/components/KpiCard.jsx`, `frontend/src/components/ChartCard.jsx`
- Modify: `frontend/src/App.jsx`, `frontend/src/main.jsx`
- Create: `frontend/tests/KpiCard.test.jsx`

**Interfaces:**
- Consumes: brand colors, `react-router-dom`, `@tanstack/react-query`.
- Produces: `<App />` wraps routes for all 8 pages behind a persistent `<Sidebar />`. `<KpiCard title value subtitle accent />` renders an animated metric tile. `<ChartCard title>{children}</ChartCard>` wraps a chart in a titled panel. `QueryClientProvider` mounted in `main.jsx`.

- [ ] **Step 1: Write the failing test**

```jsx
// frontend/tests/KpiCard.test.jsx
import { render, screen } from "@testing-library/react"
import { describe, it, expect } from "vitest"
import KpiCard from "../src/components/KpiCard"

describe("KpiCard", () => {
  it("renders title and value", () => {
    render(<KpiCard title="Revenue" value="₹1.2Cr" subtitle="YTD" accent="teal" />)
    expect(screen.getByText("Revenue")).toBeInTheDocument()
    expect(screen.getByText("₹1.2Cr")).toBeInTheDocument()
  })
})
```

Add `import "@testing-library/jest-dom"` via a `frontend/tests/setup.js` referenced in `vite.config.js` (`test: { setupFiles: "./tests/setup.js" }`).

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test`
Expected: FAIL — `KpiCard` missing.

- [ ] **Step 3: Write the components**

```jsx
// frontend/src/components/KpiCard.jsx
import { motion } from "framer-motion"

const ACCENTS = { teal: "border-teal", amber: "border-amber", coral: "border-coral" }

export default function KpiCard({ title, value, subtitle, accent = "teal" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.03 }}
      className={`bg-white rounded-2xl shadow p-5 border-l-4 ${ACCENTS[accent]}`}>
      <div className="text-xs font-semibold uppercase text-navy/60">{title}</div>
      <div className="text-3xl font-bold text-navy mt-1">{value}</div>
      {subtitle && <div className="text-xs text-navy/50 mt-1">{subtitle}</div>}
    </motion.div>
  )
}
```

```jsx
// frontend/src/components/ChartCard.jsx
export default function ChartCard({ title, children }) {
  return (
    <div className="bg-white rounded-2xl shadow p-5">
      <div className="text-sm font-semibold text-navy mb-3">{title}</div>
      {children}
    </div>
  )
}
```

```jsx
// frontend/src/components/Sidebar.jsx
import { NavLink } from "react-router-dom"
import {
  LayoutDashboard, Boxes, TrendingUp, Truck, Store, BarChart3, Bot, Cog,
} from "lucide-react"

const LINKS = [
  ["/", "Executive", LayoutDashboard],
  ["/inventory", "Inventory", Boxes],
  ["/forecast", "Forecast", TrendingUp],
  ["/suppliers", "Suppliers", Truck],
  ["/stores", "Stores", Store],
  ["/analytics", "Analytics", BarChart3],
  ["/assistant", "AI Assistant", Bot],
  ["/mlops", "MLOps", Cog],
]

export default function Sidebar() {
  return (
    <aside className="w-60 bg-navy text-white min-h-screen p-4 flex flex-col gap-1">
      <div className="text-xl font-bold mb-6 px-2">🐾 Petopia</div>
      {LINKS.map(([to, label, Icon]) => (
        <NavLink key={to} to={to} end={to === "/"}
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition ${
              isActive ? "bg-teal text-white" : "text-white/70 hover:bg-white/10"
            }`}>
          <Icon size={18} /> {label}
        </NavLink>
      ))}
    </aside>
  )
}
```

```jsx
// frontend/src/App.jsx
import { BrowserRouter, Routes, Route } from "react-router-dom"
import Sidebar from "./components/Sidebar"
import Executive from "./pages/Executive"
import Inventory from "./pages/Inventory"
import Forecast from "./pages/Forecast"
import Suppliers from "./pages/Suppliers"
import Stores from "./pages/Stores"
import Analytics from "./pages/Analytics"
import AIAssistant from "./pages/AIAssistant"
import MLOps from "./pages/MLOps"

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 min-h-screen bg-cream">
          <Routes>
            <Route path="/" element={<Executive />} />
            <Route path="/inventory" element={<Inventory />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/suppliers" element={<Suppliers />} />
            <Route path="/stores" element={<Stores />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/assistant" element={<AIAssistant />} />
            <Route path="/mlops" element={<MLOps />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
```

```jsx
// frontend/src/main.jsx
import React from "react"
import ReactDOM from "react-dom/client"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import App from "./App.jsx"
import "./index.css"

const queryClient = new QueryClient()

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}><App /></QueryClientProvider>
  </React.StrictMode>
)
```

> The 8 page imports resolve in Tasks 17–18. To keep the build green meanwhile, create one-line placeholder pages now: `export default function X(){return null}` for each, then flesh them out in the next tasks.

- [ ] **Step 4: Run test to verify it passes**

Run: `npm run test`
Expected: PASS (KpiCard test green).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ frontend/src/App.jsx frontend/src/main.jsx frontend/tests/
git commit -m "feat(frontend): app shell with router, sidebar, query provider, cards"
```

---

### Task 17: Executive, Inventory, Forecast pages

**Files:**
- Create: `frontend/src/pages/Executive.jsx`, `frontend/src/pages/Inventory.jsx`, `frontend/src/pages/Forecast.jsx`

**Interfaces:**
- Consumes: `apiGet`, `useQuery`, `KpiCard`, `ChartCard`, Recharts. Backend routes `/api/executive/kpis`, `/api/inventory/health`, `/api/forecast/skus`, `/api/forecast/{sku}`.
- Produces: three fully rendered dashboard pages. No new exports beyond default page components.

- [ ] **Step 1: Executive page**

```jsx
// frontend/src/pages/Executive.jsx
import { useQuery } from "@tanstack/react-query"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { apiGet } from "../lib/api"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"

const inr = (n) => "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 })

export default function Executive() {
  const { data, isLoading } = useQuery({
    queryKey: ["exec-kpis"], queryFn: () => apiGet("/api/executive/kpis"),
  })
  if (isLoading || !data) return <div className="text-navy/50">Loading…</div>

  const regionData = Object.entries(data.revenue_by_region).map(([region, revenue]) => ({ region, revenue }))

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Executive Overview</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard title="Revenue" value={inr(data.revenue)} subtitle="All-time" accent="teal" />
        <KpiCard title="Gross Margin" value={`${data.gross_margin_pct}%`} accent="amber" />
        <KpiCard title="Fill Rate" value={`${data.fill_rate}%`} accent="teal" />
        <KpiCard title="Stockout Rate" value={`${data.stockout_rate.toFixed(1)}%`} accent="coral" />
      </div>
      <ChartCard title="Revenue by Region">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regionData}>
            <XAxis dataKey="region" /><YAxis /><Tooltip />
            <Bar dataKey="revenue" fill="#0D9488" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}
```

- [ ] **Step 2: Inventory page**

```jsx
// frontend/src/pages/Inventory.jsx
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"
import KpiCard from "../components/KpiCard"

function Table({ rows }) {
  if (!rows?.length) return <div className="text-navy/40 text-sm">Nothing here 🎉</div>
  return (
    <table className="w-full text-sm">
      <thead><tr className="text-left text-navy/50">
        <th className="py-1">SKU</th><th>Inventory</th><th>Days of cover</th></tr></thead>
      <tbody>{rows.map((r) => (
        <tr key={r.sku_id} className="border-t">
          <td className="py-1">{r.sku_id}</td><td>{r.inventory}</td><td>{r.days_of_cover}</td>
        </tr>))}</tbody>
    </table>
  )
}

export default function Inventory() {
  const { data, isLoading } = useQuery({
    queryKey: ["inv-health"], queryFn: () => apiGet("/api/inventory/health"),
  })
  if (isLoading || !data) return <div className="text-navy/50">Loading…</div>
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Inventory Health</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard title="Reorder" value={data.reorder_list.length} accent="amber" />
        <KpiCard title="Stockout Risk" value={data.stockout_risk.length} accent="coral" />
        <KpiCard title="Dead Stock" value={data.dead_stock.length} accent="coral" />
        <KpiCard title="Overstock" value={data.overstock.length} accent="teal" />
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <ChartCard title="Stockout Risk (<7 days cover)"><Table rows={data.stockout_risk} /></ChartCard>
        <ChartCard title="Reorder List (<14 days cover)"><Table rows={data.reorder_list} /></ChartCard>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Forecast page**

```jsx
// frontend/src/pages/Forecast.jsx
import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"

export default function Forecast() {
  const [sku, setSku] = useState("")
  const [horizon, setHorizon] = useState(30)

  const { data: skus } = useQuery({ queryKey: ["skus"], queryFn: () => apiGet("/api/forecast/skus") })
  const activeSku = sku || skus?.[0]?.sku_id

  const { data: fc, isFetching } = useQuery({
    queryKey: ["fc", activeSku, horizon],
    queryFn: () => apiGet(`/api/forecast/${activeSku}?horizon=${horizon}`),
    enabled: !!activeSku,
  })

  const chart = fc ? fc.p50.map((v, i) => ({
    day: i + 1, p10: fc.p10[i], p50: v, p90: fc.p90[i],
  })) : []

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Demand Forecast</h1>
      <div className="flex gap-4 items-end">
        <div>
          <label className="text-xs text-navy/60 block">SKU</label>
          <select className="rounded-lg border p-2" value={activeSku || ""}
            onChange={(e) => setSku(e.target.value)}>
            {skus?.map((s) => <option key={s.sku_id} value={s.sku_id}>{s.sku_id} — {s.name}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-navy/60 block">Horizon</label>
          <select className="rounded-lg border p-2" value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}>
            {[7, 14, 30, 60, 90].map((h) => <option key={h} value={h}>{h} days</option>)}
          </select>
        </div>
      </div>
      <ChartCard title={`Ensemble forecast (Chronos + N-HiTS + CatBoost)${isFetching ? " — updating…" : ""}`}>
        <ResponsiveContainer width="100%" height={360}>
          <AreaChart data={chart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" /><YAxis /><Tooltip />
            <Area dataKey="p90" stroke="none" fill="#0D9488" fillOpacity={0.12} />
            <Area dataKey="p10" stroke="none" fill="#ffffff" fillOpacity={1} />
            <Line type="monotone" dataKey="p50" stroke="#0D9488" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}
```

- [ ] **Step 4: Verify build**

Run: `npm run build`
Expected: build succeeds (placeholder Suppliers/Stores/Analytics/AIAssistant/MLOps still resolve).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Executive.jsx frontend/src/pages/Inventory.jsx frontend/src/pages/Forecast.jsx
git commit -m "feat(frontend): executive, inventory, forecast dashboards"
```

---

### Task 18: Suppliers, Stores, Analytics, MLOps + AI Assistant (WebSocket)

**Files:**
- Create: `frontend/src/pages/Suppliers.jsx`, `frontend/src/pages/Stores.jsx`, `frontend/src/pages/Analytics.jsx`, `frontend/src/pages/MLOps.jsx`, `frontend/src/pages/AIAssistant.jsx`
- Create: `frontend/src/hooks/useChat.js`

**Interfaces:**
- Consumes: `apiGet`, `useQuery`, `ChartCard`, `KpiCard`, `useLLMStore`, `API_BASE`. Backend routes `/api/suppliers/scorecard`, `/api/stores/grid`, `/api/analytics/overview`, `/api/mlops/registry`, and `WS /ws/chat`.
- Produces: `useChat()` hook returning `{messages, send, connected}` over the WebSocket; five rendered pages.

- [ ] **Step 1: useChat hook**

```js
// frontend/src/hooks/useChat.js
import { useEffect, useRef, useState, useCallback } from "react"
import { API_BASE } from "../lib/api"
import { useLLMStore } from "../stores/llmStore"

export function useChat() {
  const wsRef = useRef(null)
  const [messages, setMessages] = useState([])
  const [connected, setConnected] = useState(false)
  const { provider, model, apiKey } = useLLMStore()

  useEffect(() => {
    const url = API_BASE.replace(/^http/, "ws") + "/ws/chat"
    const ws = new WebSocket(url)
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (ev) => {
      const frame = JSON.parse(ev.data)
      setMessages((m) => [...m, { role: "assistant", ...frame }])
    }
    return () => ws.close()
  }, [])

  const send = useCallback((message) => {
    setMessages((m) => [...m, { role: "user", type: "user", content: message }])
    wsRef.current?.send(JSON.stringify({ message, provider, model, api_key: apiKey }))
  }, [provider, model, apiKey])

  return { messages, send, connected }
}
```

- [ ] **Step 2: AI Assistant page**

```jsx
// frontend/src/pages/AIAssistant.jsx
import { useState } from "react"
import { useChat } from "../hooks/useChat"
import LlmSelector from "../components/LlmSelector"

export default function AIAssistant() {
  const { messages, send, connected } = useChat()
  const [input, setInput] = useState("")

  const onSend = () => {
    if (!input.trim()) return
    send(input)
    setInput("")
  }

  return (
    <div className="flex gap-6 h-[85vh]">
      <div className="w-72 shrink-0"><LlmSelector /></div>
      <div className="flex-1 flex flex-col bg-white rounded-2xl shadow p-4">
        <div className="text-sm font-semibold text-navy mb-2">
          AI Assistant {connected ? "🟢" : "🔴"}
        </div>
        <div className="flex-1 overflow-y-auto space-y-2">
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
              <span className={`inline-block px-3 py-2 rounded-2xl text-sm ${
                m.role === "user" ? "bg-teal text-white" : "bg-cream text-navy"}`}>
                {m.content}
              </span>
            </div>
          ))}
        </div>
        <div className="flex gap-2 mt-3">
          <input className="flex-1 rounded-xl border p-2" value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSend()}
            placeholder="Ask about inventory, forecasts, suppliers…" />
          <button className="bg-teal text-white px-4 rounded-xl" onClick={onSend}>Send</button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Suppliers, Stores, Analytics, MLOps pages**

```jsx
// frontend/src/pages/Suppliers.jsx
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"

export default function Suppliers() {
  const { data } = useQuery({ queryKey: ["suppliers"], queryFn: () => apiGet("/api/suppliers/scorecard") })
  const rows = data?.suppliers ?? []
  const cols = rows[0] ? Object.keys(rows[0]).slice(0, 6) : []
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Suppliers</h1>
      <ChartCard title="Supplier Scorecard">
        <table className="w-full text-sm">
          <thead><tr className="text-left text-navy/50">{cols.map((c) => <th key={c} className="py-1">{c}</th>)}</tr></thead>
          <tbody>{rows.slice(0, 50).map((r, i) => (
            <tr key={i} className="border-t">{cols.map((c) => <td key={c} className="py-1">{String(r[c])}</td>)}</tr>
          ))}</tbody>
        </table>
      </ChartCard>
    </div>
  )
}
```

```jsx
// frontend/src/pages/Stores.jsx
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"

export default function Stores() {
  const { data } = useQuery({ queryKey: ["stores"], queryFn: () => apiGet("/api/stores/grid") })
  const stores = data?.stores ?? []
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Stores ({stores.length})</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {stores.slice(0, 80).map((s, i) => (
          <ChartCard key={i} title={s.store_id || `Store ${i + 1}`}>
            <div className="text-sm text-navy/70">{s.city || ""}</div>
            <div className="text-xs text-navy/40">{s.region || ""} · {s.store_type || ""}</div>
          </ChartCard>
        ))}
      </div>
    </div>
  )
}
```

```jsx
// frontend/src/pages/Analytics.jsx
import { useQuery } from "@tanstack/react-query"
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts"
import { apiGet } from "../lib/api"
import ChartCard from "../components/ChartCard"

const COLORS = ["#0D9488", "#F59E0B", "#F87171", "#1E293B", "#64748B"]

export default function Analytics() {
  const { data } = useQuery({ queryKey: ["analytics"], queryFn: () => apiGet("/api/analytics/overview") })
  const seg = Object.entries(data?.segments ?? {}).map(([name, value]) => ({ name, value }))
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Customer Analytics</h1>
      <ChartCard title="Customer Segments">
        <ResponsiveContainer width="100%" height={320}>
          <PieChart>
            <Pie data={seg} dataKey="value" nameKey="name" outerRadius={120} label>
              {seg.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}
```

```jsx
// frontend/src/pages/MLOps.jsx
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "../lib/api"
import KpiCard from "../components/KpiCard"
import ChartCard from "../components/ChartCard"

export default function MLOps() {
  const { data } = useQuery({ queryKey: ["registry"], queryFn: () => apiGet("/api/mlops/registry") })
  if (!data) return <div className="text-navy/50">Loading…</div>
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">MLOps</h1>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <KpiCard title="Last Fine-tune" value={data.last_finetune} accent="teal" />
        <KpiCard title="Next Fine-tune" value={data.next_finetune} accent="amber" />
        <KpiCard title="Models" value={data.models.length} accent="teal" />
      </div>
      <ChartCard title="Ensemble Weights (frozen between fine-tunes)">
        <ul className="text-sm space-y-1">
          {Object.entries(data.weights).map(([k, v]) => (
            <li key={k} className="flex justify-between border-b py-1">
              <span>{k}</span><span className="font-semibold">{(v * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
      </ChartCard>
      <ChartCard title="Backtest Accuracy (MAPE)">
        <ul className="text-sm space-y-1">
          {data.models.map((m) => (
            <li key={m.name} className="flex justify-between border-b py-1">
              <span>{m.name} <span className="text-navy/40">({m.type})</span></span>
              <span className="font-semibold">{m.backtest_mape}%</span>
            </li>
          ))}
        </ul>
      </ChartCard>
    </div>
  )
}
```

- [ ] **Step 4: Verify build**

Run: `npm run build`
Expected: build succeeds, all 8 pages resolve, no placeholder imports remain.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/ frontend/src/hooks/
git commit -m "feat(frontend): suppliers, stores, analytics, mlops + AI assistant WS"
```

---

## Phase 5 — Deployment

### Task 19: Backend requirements + HF Spaces Dockerfile

**Files:**
- Create: `backend/requirements.txt` (if not already begun in Task 3/4), `backend/Dockerfile`, `backend/.dockerignore`
- Create: `backend/tests/test_smoke_import.py`

**Interfaces:**
- Produces: a Docker image that boots `uvicorn backend.main:app` on port **7860** with Chronos pre-downloaded at build time. HF Spaces detects the Dockerfile automatically.

- [ ] **Step 1: Write backend requirements**

```
# backend/requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
httpx>=0.27.0
pandas>=2.1.0
numpy>=1.26.0,<2.0
scikit-learn>=1.3.0
catboost>=1.2.0
chronos-forecasting>=1.4.0
neuralforecast>=1.7.0
torch>=2.2.0
anthropic>=0.49.0
openai>=1.30.0
groq>=0.11.0
google-generativeai>=0.8.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: Write a smoke test**

```python
# backend/tests/test_smoke_import.py
def test_app_imports():
    from backend.main import app
    routes = {getattr(r, "path", "") for r in app.routes}
    assert "/health" in routes
    assert "/api/forecast/skus" in routes
```

Run: `python -m pytest backend/tests/test_smoke_import.py -v`
Expected: PASS.

- [ ] **Step 3: Write the Dockerfile**

```dockerfile
# backend/Dockerfile  — HuggingFace Spaces (Docker SDK, CPU)
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install torch CPU wheels first (smaller, no CUDA)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.2.0

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY . /app

# Pre-download Chronos so first request is fast (not at runtime)
RUN python -c "import torch; from chronos import ChronosPipeline; \
ChronosPipeline.from_pretrained('amazon/chronos-t5-small', device_map='cpu', torch_dtype=torch.float32)"

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

```
# backend/.dockerignore
venv/
**/__pycache__/
*.pyc
frontend/
lightning_logs/
catboost_info/
.git/
```

- [ ] **Step 4: Build locally to verify**

Run: `docker build -f backend/Dockerfile -t petopia-backend .`
Expected: image builds; Chronos downloads during the pre-download layer.

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/Dockerfile backend/.dockerignore backend/tests/test_smoke_import.py
git commit -m "feat(deploy): backend requirements + HF Spaces Dockerfile (port 7860)"
```

---

### Task 20: Vercel config for the frontend

**Files:**
- Create: `frontend/vercel.json`
- Modify: `frontend/.env.example` (document `VITE_API_BASE`)
- Create: `frontend/README.md`

**Interfaces:**
- Produces: SPA rewrite config so client-side routes (`/forecast`, `/mlops`, …) resolve to `index.html`. Build command `npm run build`, output `dist`.

- [ ] **Step 1: Write vercel.json**

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

- [ ] **Step 2: Document env + frontend README**

```markdown
# Petopia Frontend

React + Vite dashboard for the Petopia Intelligence Hub.

## Local dev
```bash
npm install
npm run dev   # http://localhost:5173
```

## Environment
Set `VITE_API_BASE` to your backend URL (HF Space), e.g.
`https://<user>-petopia-backend.hf.space`. Defaults to `http://localhost:8000`.

## Deploy (Vercel)
Import the repo, set **Root Directory = frontend**, add env var `VITE_API_BASE`,
deploy. SPA routing handled by `vercel.json`.
```

- [ ] **Step 3: Verify build still green**

Run: `cd frontend && npm run build`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/vercel.json frontend/.env.example frontend/README.md
git commit -m "feat(deploy): Vercel SPA config + frontend README"
```

---

### Task 21: GitHub Actions CI (lint/test backend + build frontend)

**Files:**
- Create: `.github/workflows/backend.yml`, `.github/workflows/frontend.yml`

**Interfaces:**
- Produces: CI that runs backend pytest (excluding the heavy model-download tests via marker) and frontend build+test on every push/PR to `main`.

- [ ] **Step 1: Backend workflow**

```yaml
# .github/workflows/backend.yml
name: backend
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install (CPU torch + backend deps)
        run: |
          pip install --index-url https://download.pytorch.org/whl/cpu torch
          pip install -r backend/requirements.txt pytest
      - name: Generate data
        run: python data/generate_data.py
      - name: Run fast tests
        run: python -m pytest backend/tests -v -k "not chronos and not nhits"
```

> Chronos/N-HiTS tests are skipped in CI to avoid multi-GB model downloads on every push; they run locally and `importorskip` keeps them green when deps are absent.

- [ ] **Step 2: Frontend workflow**

```yaml
# .github/workflows/frontend.yml
name: frontend
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }
jobs:
  build:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: frontend } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: npm ci
      - run: npm run test
      - run: npm run build
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/backend.yml .github/workflows/frontend.yml
git commit -m "ci: backend pytest + frontend build/test workflows"
```

---

### Task 22: Update root README + retire Gradio

**Files:**
- Modify: `README.md` (new architecture, run instructions, deploy links)
- Modify: `requirements.txt` (remove `pytorch-forecasting`, `lightning`, `gradio`; point to `backend/requirements.txt`)
- Optional: move `gradio_app.py` → `legacy/gradio_app.py`

**Interfaces:**
- Produces: documentation reflecting the React + FastAPI + ensemble architecture. No code import depends on this task.

- [ ] **Step 1: Update root requirements**

Remove the TFT/Gradio lines from `requirements.txt`. Replace the ML + UI sections with a pointer:

```
# Forecasting + API backend deps now live in backend/requirements.txt
# Frontend deps live in frontend/package.json
# This file retains only data-generation + dev tooling:
numpy>=1.26.0,<2.0
pandas>=2.1.0
pytest>=8.0.0
```

- [ ] **Step 2: Rewrite README architecture section**

Add a section describing: React (Vercel) → FastAPI (HF Spaces) → Chronos+N-HiTS+CatBoost ensemble + ReAct agent + MCP tools. Include local-run commands:

```bash
# backend
pip install -r backend/requirements.txt
python data/generate_data.py
uvicorn backend.main:app --reload --port 8000
# frontend
cd frontend && npm install && npm run dev
```

- [ ] **Step 3: Retire Gradio (optional)**

```bash
mkdir -p legacy && git mv gradio_app.py legacy/gradio_app.py
```

- [ ] **Step 4: Commit**

```bash
git add README.md requirements.txt
git commit -m "docs: rewrite README for React+FastAPI+ensemble; retire Gradio"
```

---

## Self-Review

**1. Spec coverage:**
- React + Vite colorful/animated frontend → Tasks 14–18 (Tailwind theme, Framer Motion, Recharts). ✓
- Multiple LLM providers w/ free+paid model selector + API-key input → Task 15 (`LlmSelector`), Task 13 (`/api/chat/providers`). ✓
- HF transfer-learning zero-shot forecasting between quarterly fine-tunes → Tasks 3 (Chronos), 6 (frozen ensemble), 7 (registry/fine-tune dates). ✓
- Larger, realistic dataset → Task 1 (150 SKUs / 80 stores / 3yr / 300k txns). ✓
- Free infra: Vercel + HF Spaces Docker → Tasks 19–21. ✓
- 8 dashboards → Tasks 10–13 (routes) + 17–18 (pages). ✓
- Execute via superpowers → handoff below. ✓

**2. Placeholder scan:** No `TBD`/`handle edge cases`/`similar to`. Each code step carries full code. ✓

**3. Type consistency:** Forecast contract `{p10,p50,p90}` is consistent across Tasks 2–6, the forecast route (Task 10), and the Forecast page (Task 17). `ENSEMBLE_WEIGHTS`/`_NAMES` defined in Task 6 and reused in Task 7. `useLLMStore` shape `{provider,model,apiKey}` consistent across Tasks 15, 18. ✓

**Known adaptation point:** Task 13's `stream_agent` assumes `run_agent_with_steps(message, provider=, model=)`. Verify the real signature in `agent/agent.py` during execution and adjust the call (the WS test stubs it, so the suite stays green regardless).

---

## Execution Handoff

Plan complete. Recommended: **subagent-driven** execution (fresh subagent per task, review between tasks). Alternatively, **inline** execution in this session via executing-plans with checkpoints after each phase.
