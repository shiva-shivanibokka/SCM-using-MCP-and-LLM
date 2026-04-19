"""
Pet Store Supply Chain Intelligence — Gradio App
Designed for Hugging Face Spaces deployment.

Tabs:
  1. AI Assistant        — ReAct agent with live tool-call streaming
  2. Inventory Dashboard — At-risk SKUs, category drill-down, charts, stockout risk
  3. Analytics Dashboard — Marketing, operational, and management analytics charts
  4. Demand Forecast     — Per-SKU forecast with P10/P50/P90 and charts
  5. MLOps Monitor       — Prediction log, drift check, query log

Run locally:  python gradio_app.py
HF Spaces:    Detected automatically via module-level `demo` object
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

# Fix Windows cp1252 console encoding so emoji in tool names don't crash stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr
from dotenv import load_dotenv

import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Lazy imports (avoid import errors before data is generated)
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "huft_daily_demand.csv"

_df_cache: pd.DataFrame | None = None
# BUG-047 fix: track file modification times so caches auto-invalidate when CSVs change
_df_cache_mtime: float = 0.0
import threading as _threading

_df_cache_lock = _threading.Lock()

_transactions_cache: pd.DataFrame | None = None
_transactions_cache_mtime: float = 0.0
_products_cache: pd.DataFrame | None = None
_products_cache_mtime: float = 0.0
_stores_cache: pd.DataFrame | None = None
_stores_cache_mtime: float = 0.0
_promotions_cache: pd.DataFrame | None = None
_promotions_cache_mtime: float = 0.0
_customers_cache: pd.DataFrame | None = None
_customers_cache_mtime: float = 0.0
_cold_chain_cache: pd.DataFrame | None = None
_cold_chain_cache_mtime: float = 0.0
_supplier_perf_cache: pd.DataFrame | None = None
_supplier_perf_cache_mtime: float = 0.0


def _csv_mtime(path: Path) -> float:
    """Return file modification timestamp, or 0 if not accessible."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def get_df() -> pd.DataFrame:
    global _df_cache, _df_cache_mtime
    current_mtime = _csv_mtime(CSV_PATH)
    if _df_cache is not None and current_mtime == _df_cache_mtime:
        return _df_cache  # fast path — no lock needed
    with _df_cache_lock:  # double-checked locking
        current_mtime = _csv_mtime(CSV_PATH)
        if _df_cache is None or current_mtime != _df_cache_mtime:
            if CSV_PATH.exists():
                df_raw = pd.read_csv(CSV_PATH, parse_dates=["date"])
                # BUG-028 fix: normalize missing columns so charts never get KeyError
                if "price_inr" in df_raw.columns and "price_usd" not in df_raw.columns:
                    df_raw["price_usd"] = df_raw["price_inr"]
                elif (
                    "price_usd" in df_raw.columns and "price_inr" not in df_raw.columns
                ):
                    df_raw["price_inr"] = df_raw["price_usd"]
                for col, default in [
                    ("cost_inr", 0.0),
                    ("margin_pct", 0.0),
                    ("price_inr", 0.0),
                ]:
                    if col not in df_raw.columns:
                        df_raw[col] = default
                if "region" not in df_raw.columns:
                    df_raw["region"] = "India"
                _df_cache = df_raw
                _df_cache_mtime = current_mtime
            else:
                import sys

                sys.path.insert(0, str(BASE_DIR))
                from data.generate_data import generate

                _df_cache = generate()
    return _df_cache


# ── CSV data directory ────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
# NOTE: cache variables are declared above (lines 51-68) with mtime companions.
# Do NOT re-declare them here — the second assignment would shadow the mtime vars.


def get_transactions() -> pd.DataFrame:
    """Load huft_sales_transactions.csv with mtime-based cache invalidation."""
    global _transactions_cache, _transactions_cache_mtime
    p = DATA_DIR / "huft_sales_transactions.csv"
    mt = _csv_mtime(p)
    if _transactions_cache is None or mt != _transactions_cache_mtime:
        _transactions_cache = (
            pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
        )
        _transactions_cache_mtime = mt
    return _transactions_cache


def get_products_df() -> pd.DataFrame:
    """Load huft_products.csv with mtime-based cache invalidation."""
    global _products_cache, _products_cache_mtime
    p = DATA_DIR / "huft_products.csv"
    mt = _csv_mtime(p)
    if _products_cache is None or mt != _products_cache_mtime:
        _products_cache = pd.read_csv(p) if p.exists() else pd.DataFrame()
        _products_cache_mtime = mt
    return _products_cache


def get_stores_df() -> pd.DataFrame:
    """Load huft_stores.csv with mtime-based cache invalidation."""
    global _stores_cache, _stores_cache_mtime
    p = DATA_DIR / "huft_stores.csv"
    mt = _csv_mtime(p)
    if _stores_cache is None or mt != _stores_cache_mtime:
        _stores_cache = pd.read_csv(p) if p.exists() else pd.DataFrame()
        _stores_cache_mtime = mt
    return _stores_cache


def get_promotions_df() -> pd.DataFrame:
    """Load huft_promotions.csv with mtime-based cache invalidation."""
    global _promotions_cache, _promotions_cache_mtime
    p = DATA_DIR / "huft_promotions.csv"
    mt = _csv_mtime(p)
    if _promotions_cache is None or mt != _promotions_cache_mtime:
        _promotions_cache = (
            pd.read_csv(p, parse_dates=["start_date", "end_date"])
            if p.exists()
            else pd.DataFrame()
        )
        _promotions_cache_mtime = mt
    return _promotions_cache


def get_customers_df() -> pd.DataFrame:
    """Load huft_customers.csv with mtime-based cache invalidation."""
    global _customers_cache, _customers_cache_mtime
    p = DATA_DIR / "huft_customers.csv"
    mt = _csv_mtime(p)
    if _customers_cache is None or mt != _customers_cache_mtime:
        _customers_cache = pd.read_csv(p) if p.exists() else pd.DataFrame()
        _customers_cache_mtime = mt
    return _customers_cache


def get_cold_chain_df() -> pd.DataFrame:
    """Load huft_cold_chain.csv with mtime-based cache invalidation."""
    global _cold_chain_cache, _cold_chain_cache_mtime
    p = DATA_DIR / "huft_cold_chain.csv"
    mt = _csv_mtime(p)
    if _cold_chain_cache is None or mt != _cold_chain_cache_mtime:
        # BUG-008 fix: parse expiry_date as datetime too (consistent with server.py)
        _cold_chain_cache = (
            pd.read_csv(p, parse_dates=["date", "expiry_date"])
            if p.exists()
            else pd.DataFrame()
        )
        _cold_chain_cache_mtime = mt
    return _cold_chain_cache


def get_supplier_perf_df() -> pd.DataFrame:
    """Load huft_supplier_performance.csv with mtime-based cache invalidation."""
    global _supplier_perf_cache, _supplier_perf_cache_mtime
    p = DATA_DIR / "huft_supplier_performance.csv"
    mt = _csv_mtime(p)
    if _supplier_perf_cache is None or mt != _supplier_perf_cache_mtime:
        _supplier_perf_cache = (
            pd.read_csv(p, parse_dates=["review_month"])
            if p.exists()
            else pd.DataFrame()
        )
        _supplier_perf_cache_mtime = mt
    return _supplier_perf_cache


# ── Chart theme helpers ───────────────────────────────────────────────────────
_CHART_FONT = dict(color="#111111", family="Segoe UI, Arial, sans-serif", size=13)
_CHART_LAYOUT = dict(
    plot_bgcolor="#F8F9FA",
    paper_bgcolor="#FFFFFF",
    font=_CHART_FONT,
)


def _empty_fig(msg: str = "Data not available") -> go.Figure:
    """Return an empty figure with a centred annotation."""
    fig = go.Figure()
    fig.update_layout(
        **_CHART_LAYOUT,
        height=400,
        annotations=[
            dict(
                text=msg,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#9CA3AF"),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _fmt_inr(val: float) -> str:
    """Format a value as Indian Rupee string (e.g. ₹12,34,567)."""
    val = int(round(val))
    if val < 0:
        return f"-₹{_fmt_inr(-val)[1:]}"
    s = str(val)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    parts.reverse()
    return f"₹{','.join(parts)},{last3}"


# BUG-033 fix: declare globals at module level to prevent NameError if any
# UI component reads them before the background _init_ml thread sets them.
_ml_ready: bool = False
_ml_metrics: dict = {}


def _init_ml(
    source: str = "csv",
    mysql_creds: dict | None = None,
    pg_creds: dict | None = None,
    fine_tune: bool = False,
    force_catboost: bool = False,
) -> dict:
    """
    Train or fine-tune the TFT / CatBoost demand-forecast models.

    Parameters
    ----------
    source         : "csv" | "mysql" | "postgres"
    mysql_creds    : dict — required if source="mysql"
    pg_creds       : dict — required if source="postgres"
    fine_tune      : if True, fine-tune TFT on last 90 days (~5 min on GPU)
    force_catboost : if True, skip TFT and train CatBoost fallback directly

    Returns a metrics dict (or {"error": "..."} on failure).
    """
    global _ml_ready, _ml_metrics
    try:
        from forecasting.ml_forecast import train, load_models, get_metrics
        from forecasting.data_loader import load_training_data

        # ── On startup: try loading cached models first ───────────────────
        if source == "csv" and not fine_tune and not force_catboost and load_models():
            _ml_metrics = get_metrics()
            _ml_ready = True
            engine = _ml_metrics.get("engine", "unknown")
            logger.info(
                f"[app] {engine} loaded from cache — "
                f"MAPE={_ml_metrics.get('mape', '?')}% "
                f"trained {_ml_metrics.get('trained_at', '?')}"
            )
            return _ml_metrics

        logger.info(f"[app] Loading training data — source={source} …")
        df, source_desc = load_training_data(
            source=source,
            mysql_creds=mysql_creds,
            pg_creds=pg_creds,
            csv_path=CSV_PATH,
        )

        mode = (
            "fine-tune"
            if fine_tune
            else ("CatBoost" if force_catboost else "full TFT retrain")
        )
        logger.info(f"[app] Starting {mode} …")

        if force_catboost:
            # Directly call CatBoost training bypassing TFT
            from forecasting.ml_forecast import _train_catboost

            metrics = _train_catboost(df)
        else:
            metrics = train(df, fine_tune=fine_tune)

        metrics["data_source"] = source_desc
        _ml_metrics = metrics
        _ml_ready = True
        engine = metrics.get("engine", "Unknown")
        logger.info(
            f"[app] {engine} ready — MAPE={metrics.get('mape', '?')}% "
            f"| MAE={metrics.get('mae', '?')} | RMSE={metrics.get('rmse', '?')} "
            f"| Source: {source_desc} "
            f"| Trained: {metrics.get('trained_at', '?')}"
        )
        return metrics

    except Exception as exc:
        logger.warning(
            f"[app] CatBoost training failed ({exc}). "
            "Forecasts will use statistical fallback."
        )
        _ml_ready = False
        return {"error": str(exc)}


# Agent runner (async → sync wrapper for Gradio)


def _run_async(coro):
    """Run an async coroutine from sync context, safe inside Gradio's event loop."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


# Tab 1: AI Assistant


def _key_status(provider: str, ui_key: str) -> str:
    """Return a human-readable string describing which API key will be used."""
    from agent.agent import PROVIDER_ENV_KEYS

    env_var = PROVIDER_ENV_KEYS.get(provider.lower().strip(), "")
    ui_key = (ui_key or "").strip()
    env_key = os.getenv(env_var, "").strip()

    if ui_key:
        masked = ui_key[:6] + "..." + ui_key[-3:] if len(ui_key) > 9 else "***"
        return f"Using key typed in UI ({masked})"
    if env_key:
        masked = env_key[:6] + "..." + env_key[-3:] if len(env_key) > 9 else "***"
        return f"Using key from .env / {env_var} ({masked})"
    return f"No API key found. Type it above or set {env_var} in the .env file."


# DB credentials helpers


def _build_creds(host: str, port: str, user: str, password: str, db: str) -> dict:
    """Package UI form values into a credentials dict for MCP calls."""
    return {
        "host": (host or "").strip() or None,
        "port": int(port) if str(port).strip().isdigit() else None,
        "user": (user or "").strip() or None,
        "password": (password or "").strip() or None,
        "db": (db or "").strip() or None,
    }


def _db_status_label(host: str, db: str, tested: bool, ok: bool | None) -> str:
    """Return a one-line status string for the DB connection indicator."""
    if not host and not os.getenv("MYSQL_HOST") and not os.getenv("PG_HOST"):
        return "Not configured — fill in the fields above."
    if ok is True:
        return f"Connected to {db or 'database'} at {host}"
    if ok is False:
        return "Connection failed — check credentials."
    return f"Configured ({host or 'from .env'} / {db or 'from .env'}) — not tested yet."


def _run_test_connection(
    db_type: str, host: str, port: str, user: str, password: str, db: str
) -> str:
    """Run a DB connection test and return a formatted result string."""
    from agent.mcp_client import test_db_connection

    creds = _build_creds(host, port, user, password, db)
    # Strip None values so _resolve_*_cfg falls back to .env for missing fields
    creds_clean = {k: v for k, v in creds.items() if v is not None}
    result = _run_async(test_db_connection(db_type, creds_clean))
    if result.get("ok"):
        d = result.get("details", {})
        lines = [f"Connected successfully to {db_type.upper()}"]
        if d.get("host"):
            lines.append(f"  Host:     {d['host']}:{d.get('port', '')}")
        if d.get("database"):
            lines.append(f"  Database: {d['database']}")
        if d.get("version"):
            lines.append(f"  Version:  {d['version']}")
        return "\n".join(lines)
    return f"Connection failed: {result.get('message', 'Unknown error')}"


# ---------------------------------------------------------------------------
# Local / Cloud toggle logic
# ---------------------------------------------------------------------------

# Default values for "Local" mode — pulled from .env at startup
_LOCAL_DEFAULTS = {
    "mysql": {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": os.getenv("MYSQL_PORT", "3306"),
        "user": os.getenv("MYSQL_USER", "root"),
        "db": os.getenv("MYSQL_DB", "pet_store_scm"),
    },
    "pg": {
        "host": os.getenv("PG_HOST", "localhost"),
        "port": os.getenv("PG_PORT", "5432"),
        "user": os.getenv("PG_USER", "postgres"),
        "db": os.getenv("PG_DB", "pet_store_scm"),
    },
}

# Cloud credentials — pulled from MYSQL_CLOUD_* / PG_CLOUD_* in .env
_CLOUD_DEFAULTS = {
    "mysql": {
        "host": os.getenv("MYSQL_CLOUD_HOST", ""),
        "port": os.getenv("MYSQL_CLOUD_PORT", ""),
        "user": os.getenv("MYSQL_CLOUD_USER", ""),
        "db": os.getenv("MYSQL_CLOUD_DB", "railway"),
    },
    "pg": {
        "host": os.getenv("PG_CLOUD_HOST", ""),
        "port": os.getenv("PG_CLOUD_PORT", ""),
        "user": os.getenv("PG_CLOUD_USER", ""),
        "db": os.getenv("PG_CLOUD_DB", "railway"),
    },
}

# Per-provider cloud instructions shown as helper text in the UI
_CLOUD_PROVIDER_HINTS = {
    "Railway": {
        "how_to_get": (
            "1. Go to railway.app and create a free account\n"
            "2. Click 'New Project' → 'Provision MySQL' (or PostgreSQL)\n"
            "3. Click the service → 'Connect' tab\n"
            "4. Copy the HOST, PORT, USER, PASSWORD, DATABASE values from there\n"
            "5. Paste them into the fields below and click Test Connection"
        ),
        "mysql_port": "varies",
        "pg_port": "varies",
        "ssl_needed": False,
        "free_tier": "Yes — $5 credit/month, no card needed",
    },
    "Supabase (PostgreSQL only)": {
        "how_to_get": (
            "1. Go to supabase.com and create a free account\n"
            "2. Create a new project — note the database password you set\n"
            "3. Go to Project Settings → Database\n"
            "4. Under 'Connection parameters' copy Host, Port (5432), User, Database\n"
            "5. Paste them into the PostgreSQL fields below and click Test Connection\n"
            "Note: Supabase is PostgreSQL only. Use a different provider for MySQL."
        ),
        "mysql_port": "N/A",
        "pg_port": "5432",
        "ssl_needed": True,
        "free_tier": "Yes — 500 MB, 2 projects free",
    },
    "Render": {
        "how_to_get": (
            "1. Go to render.com and create a free account\n"
            "2. Click 'New' → 'PostgreSQL' (free) or 'MySQL' (paid)\n"
            "3. After creation, open the service and copy the 'External Database URL'\n"
            "4. Parse out host, port, user, password, database from the URL\n"
            "   Format: postgresql://user:password@host:port/database\n"
            "5. Paste into the fields below and click Test Connection"
        ),
        "mysql_port": "3306",
        "pg_port": "5432",
        "ssl_needed": True,
        "free_tier": "PostgreSQL free, MySQL requires paid plan",
    },
    "Aiven": {
        "how_to_get": (
            "1. Go to aiven.io and create a free account\n"
            "2. Create a new service — choose MySQL or PostgreSQL, pick the free trial\n"
            "3. Once running, click the service and go to the 'Overview' tab\n"
            "4. Copy HOST, PORT, USER, PASSWORD, DATABASE from the connection details\n"
            "5. Paste into the fields below and click Test Connection\n"
            "Note: Aiven requires SSL — the app handles this automatically."
        ),
        "mysql_port": "varies",
        "pg_port": "varies",
        "ssl_needed": True,
        "free_tier": "One free trial service, no card needed",
    },
    "Neon (PostgreSQL only)": {
        "how_to_get": (
            "1. Go to neon.tech and create a free account\n"
            "2. Create a new project — a database is created automatically\n"
            "3. Go to Dashboard → Connection Details\n"
            "4. Copy the host, user, password, and database name\n"
            "5. Port is always 5432\n"
            "6. Paste into the PostgreSQL fields below and click Test Connection\n"
            "Note: Neon is PostgreSQL only."
        ),
        "mysql_port": "N/A",
        "pg_port": "5432",
        "ssl_needed": True,
        "free_tier": "Yes — generous free tier, no card needed",
    },
    "AWS RDS": {
        "how_to_get": (
            "1. Go to AWS Console → RDS → Create Database\n"
            "2. Choose MySQL or PostgreSQL, select 'Free tier'\n"
            "3. Set a master username and password\n"
            "4. After creation, go to the RDS instance and copy the Endpoint (host)\n"
            "5. Port is 3306 (MySQL) or 5432 (PostgreSQL)\n"
            "6. Make sure the security group allows inbound traffic on that port\n"
            "7. Paste into the fields below and click Test Connection"
        ),
        "mysql_port": "3306",
        "pg_port": "5432",
        "ssl_needed": True,
        "free_tier": "12 months free tier (new AWS accounts only)",
    },
    "PlanetScale (MySQL only)": {
        "how_to_get": (
            "1. Go to planetscale.com and create a free account\n"
            "2. Create a new database and choose a region\n"
            "3. Go to the database → Connect → 'Connect with' → select 'General'\n"
            "4. Copy host, username, and password from the connection string\n"
            "5. Database name is the name you gave your PlanetScale database\n"
            "6. Port is 3306\n"
            "7. Paste into the MySQL fields below and click Test Connection\n"
            "Note: PlanetScale is MySQL only."
        ),
        "mysql_port": "3306",
        "pg_port": "N/A",
        "ssl_needed": True,
        "free_tier": "Yes — 5 GB free, no card needed",
    },
}


def _apply_local_preset() -> tuple:
    """Return field values for Local mode."""
    m = _LOCAL_DEFAULTS["mysql"]
    p = _LOCAL_DEFAULTS["pg"]
    return (
        m["host"],
        m["port"],
        m["user"],
        m["db"],
        p["host"],
        p["port"],
        p["user"],
        p["db"],
        "Not tested yet.",
        "Not tested yet.",
        "",  # provider hint box
    )


def _apply_cloud_preset(provider: str) -> tuple:
    """Return field values for Cloud mode — pre-fills from MYSQL_CLOUD_* / PG_CLOUD_* .env vars."""
    hint_data = _CLOUD_PROVIDER_HINTS.get(provider, {})
    hint_text = hint_data.get(
        "how_to_get", "Select a provider above to see instructions."
    )

    # Use cloud .env values if set, otherwise fall back to empty / hint port
    cm = _CLOUD_DEFAULTS["mysql"]
    cp = _CLOUD_DEFAULTS["pg"]

    mysql_port_hint = (
        hint_data.get("mysql_port", "3306")
        if hint_data.get("mysql_port") not in ("N/A", "varies")
        else ""
    )
    pg_port_hint = (
        hint_data.get("pg_port", "5432")
        if hint_data.get("pg_port") not in ("N/A", "varies")
        else ""
    )

    return (
        cm["host"],  # mysql host  (from MYSQL_CLOUD_HOST or "")
        cm["port"] or mysql_port_hint,  # mysql port  (from MYSQL_CLOUD_PORT or hint)
        cm["user"],  # mysql user  (from MYSQL_CLOUD_USER or "")
        cm["db"],  # mysql db    (from MYSQL_CLOUD_DB)
        cp["host"],  # pg host     (from PG_CLOUD_HOST or "")
        cp["port"] or pg_port_hint,  # pg port     (from PG_CLOUD_PORT or hint)
        cp["user"],  # pg user     (from PG_CLOUD_USER or "")
        cp["db"],  # pg db       (from PG_CLOUD_DB)
        "Not tested yet.",
        "Not tested yet.",
        hint_text,
    )


def _on_mode_change(mode: str, provider: str) -> tuple:
    """Called when the Local/Cloud radio or provider dropdown changes."""
    if mode == "Local":
        return _apply_local_preset()
    else:
        return _apply_cloud_preset(provider)


def build_assistant_tab(
    provider_dd,
    model_dd,
    api_key_box,
    key_status_box,
    mysql_host,
    mysql_port,
    mysql_user,
    mysql_password,
    mysql_db,
    pg_host,
    pg_port,
    pg_user,
    pg_password,
    pg_db,
):
    with gr.TabItem("AI Assistant"):
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#4285F4 0%,#1a73e8 100%);
                    border-radius:12px; padding:20px 26px; margin-bottom:18px;
                    box-shadow:0 4px 16px rgba(66,133,244,0.25);">
            <div style="color:#fff; font-size:1.35rem; font-weight:800; margin-bottom:6px;
                        letter-spacing:-0.01em;">
                Pet Store Supply Chain Assistant
            </div>
            <div style="color:rgba(255,255,255,0.9); font-size:0.95rem; line-height:1.6;">
                Ask anything about inventory, demand forecasts, suppliers, or policies.
                The ReAct agent reasons step-by-step and calls live database tools to answer.
            </div>
        </div>
        """)

        _WELCOME = (
            "Hi! I'm your **Pet Store Supply Chain** assistant, powered by a ReAct "
            "agent with live MCP database tools.\n\n"
            "Here are some things you can ask:\n\n"
            "- Which SKUs are critically low on stock right now?\n"
            "- What is the 30-day demand forecast for premium dog food?\n"
            "- Which suppliers have lead times over 14 days?\n"
            "- Show me inventory risk for the cat food category.\n"
            "- Recommend reorder quantities for at-risk items.\n\n"
            "*Type your question below and press **Enter** to send.*"
        )

        chatbot = gr.Chatbot(
            label="",
            show_label=False,
            height=580,
            value=[{"role": "assistant", "content": _WELCOME}],
            elem_classes=["full-width-chat"],
        )

        msg_box = gr.Textbox(
            placeholder="Ask about inventory, forecasts, suppliers... (press Enter to send)",
            label="Your Question",
            lines=1,  # lines=1 → plain Enter submits; lines>1 requires Shift+Enter
        )
        clear_btn = gr.Button("Clear Chat", variant="secondary")

        def chat(
            user_msg,
            history,
            provider,
            model,
            api_key,
            m_host,
            m_port,
            m_user,
            m_pw,
            m_db,
            p_host,
            p_port,
            p_user,
            p_pw,
            p_db,
        ):
            import queue as _queue
            import threading as _threading

            if not user_msg.strip():
                yield history or [], ""
                return

            mysql_creds = {
                k: v
                for k, v in _build_creds(m_host, m_port, m_user, m_pw, m_db).items()
                if v is not None
            }
            pg_creds = {
                k: v
                for k, v in _build_creds(p_host, p_port, p_user, p_pw, p_db).items()
                if v is not None
            }

            from mcp_server.server import set_session_creds

            set_session_creds(mysql_creds, pg_creds)

            history = list(history or [])
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": "Starting agent..."})
            yield history, ""

            from agent.agent import run_agent_with_steps

            # ── Stream thinking steps via a queue so we can yield live ────────
            step_q: _queue.Queue = _queue.Queue()

            def _worker():
                async def _collect():
                    async for step in run_agent_with_steps(
                        user_query=user_msg,
                        provider=provider.lower().strip(),
                        model=model,
                        api_key=api_key,
                    ):
                        step_q.put(step)
                    step_q.put(None)  # sentinel — done

                asyncio.run(_collect())

            t = _threading.Thread(target=_worker, daemon=True)
            t.start()

            thinking_lines: list[str] = []
            final_answer = ""
            tools_called: list[str] = []
            _t_start = _threading.Event()
            import time as _time

            _wall_start = _time.monotonic()

            while True:
                try:
                    step = step_q.get(timeout=300)
                except _queue.Empty:
                    final_answer = "Agent timed out after 300 s."
                    break

                if step is None:
                    break

                if step["type"] == "thinking":
                    # LLM's own reasoning text — show this live
                    text = step["text"].strip()
                    if text:
                        thinking_lines.append(text)
                        display = "*Thinking...*\n\n" + "\n\n".join(
                            f"> {line}" for line in thinking_lines
                        )
                        history[-1] = {"role": "assistant", "content": display}
                        yield history, ""

                elif step["type"] == "answer":
                    final_answer = step["text"]

                elif step["type"] == "error":
                    final_answer = f"Error: {step['text']}"

                elif step["type"] == "tool_start":
                    # Track which tools were called for the MLOps query log (M-07 fix)
                    tool_name = step.get("tool", "")
                    if tool_name and tool_name not in tools_called:
                        tools_called.append(tool_name)

                # tool_result is intentionally ignored in the UI.

            t.join(timeout=5)

            # Log this query to the MLOps monitor (M-07 fix)
            try:
                from mlops.monitor import log_query as _log_query

                _log_query(
                    user_query=user_msg,
                    provider=provider.lower().strip(),
                    model=model,
                    tools_called=tools_called,
                    duration_ms=int((_time.monotonic() - _wall_start) * 1000),
                )
            except Exception:
                pass  # monitoring must never break the chat response

            history[-1] = {
                "role": "assistant",
                "content": final_answer or "No response.",
            }
            yield history, ""

        _db_inputs = [
            mysql_host,
            mysql_port,
            mysql_user,
            mysql_password,
            mysql_db,
            pg_host,
            pg_port,
            pg_user,
            pg_password,
            pg_db,
        ]

        msg_box.submit(
            chat,
            inputs=[msg_box, chatbot, provider_dd, model_dd, api_key_box] + _db_inputs,
            outputs=[chatbot, msg_box],
        )
        clear_btn.click(
            lambda: ([{"role": "assistant", "content": _WELCOME}], ""),
            outputs=[chatbot, msg_box],
        )

        # Update key status when provider or key changes
        for trigger in [provider_dd, api_key_box]:
            trigger.change(
                _key_status,
                inputs=[provider_dd, api_key_box],
                outputs=[key_status_box],
            )


# ── Chart Builders ───────────────────────────────────────────────────────────


def build_inventory_heatmap(category_filter: str = "All") -> go.Figure:
    """
    Inventory Health — horizontal bar chart per SKU.
    Each row = one SKU (ID + short name), bar length = days_of_supply, colour = risk level.
    Separate traces per risk class so the legend is meaningful.
    """
    try:
        df = get_df()
        if df.empty:
            return _empty_fig("No inventory data available")

        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].copy()
        cutoff = latest_date - pd.Timedelta(days=30)
        recent = (
            df[df["date"] >= cutoff]
            .groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "avg_daily_demand"})
        )
        merged = latest.merge(recent, on="sku_id")
        merged["days_of_supply"] = (
            (merged["inventory"] / merged["avg_daily_demand"].replace(0, np.nan))
            .fillna(0)
            .round(1)
        )

        def _risk(dos, lt):
            if dos < lt:
                return "CRITICAL"
            if dos < 2 * lt:
                return "WARNING"
            return "OK"

        merged["risk"] = merged.apply(
            lambda r: _risk(r["days_of_supply"], r["lead_time_days"]), axis=1
        )

        if category_filter != "All":
            merged = merged[merged["category"] == category_filter]

        if merged.empty:
            return _empty_fig(f"No data for category: {category_filter}")

        merged = merged.sort_values("days_of_supply", ascending=True).reset_index(
            drop=True
        )

        # Build readable Y-axis label: "SKU_ID — Short Name"
        if "name" in merged.columns:
            merged["label"] = merged["sku_id"] + " — " + merged["name"].str[:20]
        else:
            merged["label"] = merged["sku_id"]

        color_map = {"CRITICAL": "#EA4335", "WARNING": "#F9AB00", "OK": "#34A853"}
        risk_order = ["CRITICAL", "WARNING", "OK"]

        fig = go.Figure()
        for risk in risk_order:
            sub = merged[merged["risk"] == risk]
            if sub.empty:
                continue
            fig.add_trace(
                go.Bar(
                    name=risk,
                    x=sub["days_of_supply"],
                    y=sub["label"],
                    orientation="h",
                    marker_color=color_map[risk],
                    marker_opacity=0.88,
                    text=[f"{d:.0f}d" for d in sub["days_of_supply"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    customdata=sub[
                        ["inventory", "avg_daily_demand", "lead_time_days", "category"]
                    ].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Days of Supply: <b>%{x:.1f}d</b><br>"
                        "Lead Time: <b>%{customdata[2]:.0f}d</b> "
                        "(critical if DoS < this, warning if < 2×this)<br>"
                        "Inventory: %{customdata[0]:,} units<br>"
                        "Avg Daily Demand: %{customdata[1]:.1f}/day<br>"
                        "Category: %{customdata[3]}<extra></extra>"
                    ),
                )
            )

        # Per-SKU lead-time markers — ◆ at critical threshold, ▲ at warning threshold
        fig.add_trace(
            go.Scatter(
                x=merged["lead_time_days"],
                y=merged["label"],
                mode="markers",
                name="Lead Time (critical threshold)",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color="#C5221F",
                    line=dict(color="#fff", width=1),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>Critical threshold: <b>%{x}d</b> (= lead time)<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged["lead_time_days"] * 2,
                y=merged["label"],
                mode="markers",
                name="2× Lead Time (warning threshold)",
                marker=dict(
                    symbol="triangle-right",
                    size=8,
                    color="#F9AB00",
                    line=dict(color="#fff", width=1),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>Warning threshold: <b>%{x}d</b> (= 2 × lead time)<extra></extra>"
                ),
            )
        )

        crit_count = int((merged["risk"] == "CRITICAL").sum())
        warn_count = int((merged["risk"] == "WARNING").sum())
        scope = category_filter if category_filter != "All" else "All Categories"
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    f"Inventory Health — {scope}<br>"
                    f"<sup style='color:#666'>{crit_count} Critical &nbsp;|&nbsp; "
                    f"{warn_count} Warning &nbsp;|&nbsp; "
                    f"{len(merged) - crit_count - warn_count} OK &nbsp;(of {len(merged)} SKUs) "
                    f"&nbsp;·&nbsp; ◆ = each SKU's own lead-time threshold</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis_title="Days of Supply remaining",
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            height=max(440, len(merged) * 22 + 160),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(t=90, b=60, l=190, r=100),
            barmode="overlay",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error building heatmap: {exc}")


def build_sales_by_channel(
    period_days: int = 90, store_id: str | None = None
) -> go.Figure:
    """
    Sales by Channel Over Time — Stacked area chart.
    Groups huft_sales_transactions.csv by week and channel.
    """
    try:
        txn = get_transactions()
        if store_id:
            txn = txn[txn["store_id"] == store_id]
        if txn.empty:
            return _empty_fig("Sales transaction data not available")

        cutoff = txn["date"].max() - pd.Timedelta(days=period_days)
        txn = txn[txn["date"] >= cutoff].copy()
        txn["week"] = txn["date"].dt.to_period("W").dt.start_time
        weekly = txn.groupby(["week", "channel"])["net_revenue_inr"].sum().reset_index()

        channels = weekly["channel"].unique().tolist()
        channel_colors = {"Online": "#4285F4", "Offline": "#34A853", "App": "#EA4335"}

        fig = go.Figure()
        for ch in sorted(channels):
            sub = weekly[weekly["channel"] == ch].sort_values("week")
            fig.add_trace(
                go.Scatter(
                    x=sub["week"],
                    y=sub["net_revenue_inr"],
                    name=ch,
                    mode="lines",
                    stackgroup="one",
                    fillcolor=channel_colors.get(ch, "#9CA3AF"),
                    line=dict(color=channel_colors.get(ch, "#9CA3AF"), width=2),
                    hovertemplate=f"<b>{ch}</b><br>Week: %{{x|%b %d}}<br>Revenue: ₹%{{y:,.0f}}<extra></extra>",
                )
            )

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=f"Sales by Channel — Last {period_days} Days (Weekly)",
                font=dict(size=16, color="#4285F4"),
            ),
            xaxis_title="Week",
            yaxis_title="Net Revenue (₹)",
            height=440,
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(t=80, b=60),
            hovermode="x unified",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_brand_performance_bubble(
    category_filter: str = "All",
    channel_filter: str = "All",
    period_days: int = 365,
    store_id: str | None = None,
) -> go.Figure:
    """
    Brand Performance — dual horizontal bar chart.
    Left bar = Revenue (₹), Right bar = Gross Margin %.
    Brands sorted by revenue descending. Color = Private Label vs Third Party.
     Filters: category, channel, period, store_id.
    """
    try:
        txn = get_transactions()
        if store_id:
            txn = txn[txn["store_id"] == store_id]
        products = get_products_df()
        if txn.empty or products.empty:
            return _empty_fig("Transaction or product data not available")

        # Apply period filter
        cutoff = txn["date"].max() - pd.Timedelta(days=period_days)
        txn = txn[txn["date"] >= cutoff].copy()

        prod_slim = products[
            ["sku_id", "brand_type", "brand", "margin_pct"]
        ].drop_duplicates("sku_id")
        merged = txn.merge(prod_slim, on="sku_id", how="left", suffixes=("", "_prod"))

        if category_filter != "All":
            merged = merged[merged["category"] == category_filter]
        if channel_filter != "All":
            merged = merged[merged["channel"] == channel_filter]

        if merged.empty:
            return _empty_fig(f"No data for {category_filter} / {channel_filter}")

        # BUG-009 fix: drop rows where brand is NaN before groupby to avoid a
        # spurious "NaN" brand row in the output.
        merged = merged.dropna(subset=["brand"])
        agg = (
            merged.groupby(["brand", "brand_type"])
            .agg(
                revenue=("net_revenue_inr", "sum"),
                margin=("gross_margin_inr", "sum"),
                units=("quantity", "sum"),
            )
            .reset_index()
        )
        agg["margin_pct_actual"] = (
            agg["margin"] / agg["revenue"].replace(0, np.nan) * 100
        ).fillna(0)
        agg = agg[agg["revenue"] > 0].sort_values("revenue", ascending=True).tail(20)

        color_map = {"Private Label": "#4285F4", "Third Party": "#EA4335"}
        bar_colors = agg["brand_type"].map(color_map).fillna("#9CA3AF").tolist()

        fig = go.Figure()

        # Revenue bars
        fig.add_trace(
            go.Bar(
                name="Revenue (₹)",
                y=agg["brand"],
                x=agg["revenue"],
                orientation="h",
                marker_color=bar_colors,
                marker_opacity=0.85,
                text=[
                    f"₹{v / 1e6:.2f}M | {m:.0f}% margin"
                    for v, m in zip(agg["revenue"], agg["margin_pct_actual"])
                ],
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>",
                xaxis="x1",
            )
        )

        # Margin % dots on secondary axis
        fig.add_trace(
            go.Scatter(
                name="Margin %",
                y=agg["brand"],
                x=agg["margin_pct_actual"],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color="#34A853",
                    line=dict(width=1.5, color="#137333"),
                ),
                hovertemplate="<b>%{y}</b><br>Margin: %{x:.1f}%<extra></extra>",
                xaxis="x2",
            )
        )

        # Add invisible traces for legend (brand type)
        for btype, col in color_map.items():
            fig.add_trace(
                go.Bar(
                    name=btype,
                    x=[None],
                    y=[None],
                    orientation="h",
                    marker_color=col,
                    showlegend=True,
                )
            )

        n = len(agg)
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=f"Brand Performance — {category_filter}"
                + (f" | {channel_filter}" if channel_filter != "All" else ""),
                font=dict(size=16, color="#4285F4"),
            ),
            xaxis=dict(title="Total Revenue (₹)", domain=[0, 0.78], showgrid=True),
            xaxis2=dict(
                title="Gross Margin %",
                overlaying="x",
                side="top",
                range=[0, 100],
                domain=[0, 0.78],
                showgrid=False,
            ),
            yaxis=dict(tickfont=dict(size=11)),
            height=max(420, n * 34 + 120),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
            margin=dict(t=90, b=60, l=160, r=120),
            barmode="overlay",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_category_revenue_heatmap(
    category_filter: str = "All",
    channel_filter: str = "All",
    period_days: int = 365,
    store_id: str | None = None,
) -> go.Figure:
    """
    Category Monthly Revenue Trend — multi-line chart.
    One line per category showing monthly revenue over time.
    Far more readable than a heatmap: peaks, troughs, and trends are immediately visible.
    Filters: category (single line), channel, period, store_id.
    """
    try:
        txn = get_transactions()
        if store_id:
            txn = txn[txn["store_id"] == store_id]
        if txn.empty:
            return _empty_fig("Sales transaction data not available")

        cutoff = txn["date"].max() - pd.Timedelta(days=period_days)
        txn = txn[txn["date"] >= cutoff].copy()
        if channel_filter != "All":
            txn = txn[txn["channel"] == channel_filter]
        if category_filter != "All":
            txn = txn[txn["category"] == category_filter]

        if txn.empty:
            return _empty_fig("No data for selected filters")

        # Monthly aggregation per category
        txn["month_dt"] = txn["date"].dt.to_period("M").dt.to_timestamp()
        monthly = (
            txn.groupby(["category", "month_dt"])
            .agg(revenue=("net_revenue_inr", "sum"), units=("quantity", "sum"))
            .reset_index()
        )

        if monthly.empty:
            return _empty_fig("No data for selected filters")

        categories = sorted(monthly["category"].unique())
        # Distinct colors — up to 14 categories
        palette = [
            "#4285F4",
            "#EA4335",
            "#34A853",
            "#FBBC05",
            "#7C3AED",
            "#0891B2",
            "#F97316",
            "#EC4899",
            "#14B8A6",
            "#8B5CF6",
            "#06B6D4",
            "#84CC16",
            "#F59E0B",
            "#EF4444",
        ]

        fig = go.Figure()

        for i, cat in enumerate(categories):
            cat_df = monthly[monthly["category"] == cat].sort_values("month_dt")
            color = palette[i % len(palette)]

            # Add shaded area under each line for easier reading
            fig.add_trace(
                go.Scatter(
                    x=cat_df["month_dt"],
                    y=cat_df["revenue"],
                    name=cat,
                    mode="lines+markers",
                    line=dict(color=color, width=2.2),
                    marker=dict(size=5, color=color),
                    fill="tozeroy" if len(categories) == 1 else "none",
                    fillcolor=color.replace(")", ",0.08)").replace("rgb(", "rgba(")
                    if "rgb" in color
                    else color + "14",
                    hovertemplate=(
                        f"<b>{cat}</b><br>"
                        "Month: %{x|%b %Y}<br>"
                        "Revenue: ₹%{y:,.0f}<br>"
                        "Units: %{customdata:,}<extra></extra>"
                    ),
                    customdata=cat_df["units"],
                )
            )

        # Add festival period annotations as vertical bands
        # Covers 2023–2026 so the chart remains accurate for current deployments
        festivals = [
            ("Diwali", "2023-10-15", "2023-11-15"),
            ("New Year", "2024-01-01", "2024-01-07"),
            ("Holi", "2024-03-20", "2024-03-26"),
            ("Independence Day", "2024-08-12", "2024-08-16"),
            ("Diwali", "2024-10-25", "2024-11-05"),
            ("New Year", "2025-01-01", "2025-01-07"),
            ("Holi", "2025-03-14", "2025-03-16"),
            ("Independence Day", "2025-08-11", "2025-08-15"),
            ("Diwali", "2025-10-17", "2025-10-25"),
            ("New Year", "2026-01-01", "2026-01-07"),
        ]
        for fname, fs, fe in festivals:
            fstart = pd.Timestamp(fs)
            fend = pd.Timestamp(fe)
            if fstart >= cutoff:
                fig.add_vrect(
                    x0=fstart,
                    x1=fend,
                    fillcolor="#FBBC05",
                    opacity=0.10,
                    line_width=0,
                    annotation_text=fname,
                    annotation_position="top left",
                    annotation_font_size=9,
                    annotation_font_color="#92400E",
                )

        subtitle = (
            (category_filter if category_filter != "All" else "All Categories")
            + (f" | {channel_filter}" if channel_filter != "All" else "")
            + f" | Last {period_days} days"
        )
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=f"Monthly Revenue by Category — {subtitle}",
                font=dict(size=15, color="#4285F4"),
            ),
            xaxis=dict(
                title="Month",
                tickformat="%b %Y",
                tickangle=35,
                tickfont=dict(size=10),
            ),
            yaxis=dict(title="Monthly Revenue (₹)", tickformat=",.0f"),
            height=480,
            legend=dict(
                orientation="h",
                y=-0.22,
                x=0,
                font=dict(size=10),
                itemsizing="constant",
            ),
            hovermode="x unified",
            margin=dict(t=80, b=120, l=70, r=40),
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_promotion_impact(
    promo_filter: str = "All",
    category_filter: str = "All",
    channel_filter: str = "All",
    store_id: str | None = None,
) -> go.Figure:
    """
    Promotion Demand Lift Chart.
    Shows % lift in daily demand During vs 7-day baseline Before each promotion.
    Positive = promotion boosted demand. Negative = demand fell (e.g. donation drive).
    Sorted by lift % so best-performing promotions are immediately visible.
    Also shows 14-day post-promo retention: did demand hold after the sale ended?
    """
    try:
        promos = get_promotions_df()
        demand_df = get_df()
        if promos.empty or demand_df.empty:
            return _empty_fig("Promotion or demand data not available")

        # BUG-015/016 fix: channel filter requires transactions CSV (demand CSV has no
        # channel column). When channel filter is active, aggregate transactions to
        # daily demand per category; otherwise use the faster demand CSV.
        if channel_filter != "All" or store_id:
            txn = get_transactions()
            if not txn.empty:
                if store_id:
                    txn = txn[txn["store_id"] == store_id]
                if channel_filter != "All" and "channel" in txn.columns:
                    txn = txn[txn["channel"] == channel_filter]
            if not txn.empty and "channel" in txn.columns:
                txn_filt = txn.copy()
                if category_filter != "All":
                    txn_filt = txn_filt[txn_filt["category"] == category_filter]
                # Aggregate to daily total demand (quantity) per date+category
                demand_filt = (
                    txn_filt.groupby(["date", "sku_id", "category"])["quantity"]
                    .sum()
                    .reset_index()
                    .rename(columns={"quantity": "demand"})
                )
            else:
                demand_filt = demand_df.copy()
                if category_filter != "All":
                    demand_filt = demand_filt[
                        demand_filt["category"] == category_filter
                    ]
        else:
            demand_filt = demand_df.copy()
            if category_filter != "All":
                demand_filt = demand_filt[demand_filt["category"] == category_filter]

        if promo_filter != "All":
            # Single promotion — time-series with before/during/after bands
            row = promos[promos["name"] == promo_filter]
            if row.empty:
                return _empty_fig(f"Promotion '{promo_filter}' not found")
            row = row.iloc[0]
            start = pd.to_datetime(row["start_date"])
            end = pd.to_datetime(row["end_date"])
            w_start = start - pd.Timedelta(days=14)
            w_end = end + pd.Timedelta(days=21)

            window = (
                demand_filt[
                    (demand_filt["date"] >= w_start) & (demand_filt["date"] <= w_end)
                ]
                .groupby("date")["demand"]
                .sum()
                .reset_index()
            )
            # BUG-22 fix: use per-day total demand (not per-row mean) as baseline.
            # The channel-filter path produces one row per SKU per day, so .mean()
            # gives avg per-SKU demand, not total daily demand. .sum().mean() gives
            # the correct mean total daily demand for the pre-promo window.
            _baseline_window = demand_filt[
                (demand_filt["date"] >= start - pd.Timedelta(days=7))
                & (demand_filt["date"] < start)
            ]
            _baseline_daily = _baseline_window.groupby("date")["demand"].sum()
            _baseline_raw = (
                _baseline_daily.mean() if not _baseline_daily.empty else float("nan")
            )
            baseline = (
                float(_baseline_raw)
                if pd.notna(_baseline_raw) and float(_baseline_raw) > 0
                else 1.0
            )

            ch_label = (
                f" | Channel: {channel_filter}" if channel_filter != "All" else ""
            )
            fig = go.Figure()
            # Baseline reference line
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color="#9E9E9E",
                annotation_text=f"7-day baseline: {baseline:.0f}",
                annotation_font_size=10,
            )
            fig.add_trace(
                go.Scatter(
                    x=window["date"],
                    y=window["demand"],
                    mode="lines+markers",
                    name="Daily Demand",
                    line=dict(color="#4285F4", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(66,133,244,0.08)",
                )
            )
            fig.add_vrect(
                x0=str(start)[:10],
                x1=str(end)[:10],
                fillcolor="#FBBC05",
                opacity=0.15,
                line_width=0,
            )
            fig.add_annotation(
                x=str(start)[:10],
                y=1,
                yref="paper",
                text=f"🎯 {str(row['name'])[:20]}",
                showarrow=False,
                font=dict(size=10, color="#92400E"),
                xanchor="left",
            )
            fig.update_layout(
                **_CHART_LAYOUT,
                title=dict(
                    text=f"Demand During: {promo_filter}{ch_label}",
                    font=dict(size=15, color="#4285F4"),
                ),
                xaxis_title="Date",
                yaxis_title="Daily Demand (units)",
                height=420,
                margin=dict(t=80, b=60),
            )
            return fig

        # All promotions — Demand Lift % horizontal bar
        results = []
        for _, row in promos.iterrows():
            start = pd.to_datetime(row["start_date"])
            end = pd.to_datetime(row["end_date"])
            before_mask = (demand_filt["date"] >= start - pd.Timedelta(days=7)) & (
                demand_filt["date"] < start
            )
            during_mask = (demand_filt["date"] >= start) & (demand_filt["date"] <= end)
            after_mask = (demand_filt["date"] > end) & (
                demand_filt["date"] <= end + pd.Timedelta(days=14)
            )

            # BUG-22 fix: use mean of daily totals, not mean of per-row values
            before = float(
                demand_filt[before_mask].groupby("date")["demand"].sum().mean() or 0
            )
            during = float(
                demand_filt[during_mask].groupby("date")["demand"].sum().mean() or 0
            )
            after = float(
                demand_filt[after_mask].groupby("date")["demand"].sum().mean() or 0
            )

            if before < 1:
                continue
            lift_during = (during - before) / before * 100
            lift_after = (after - before) / before * 100

            results.append(
                {
                    "name": str(row["name"]),
                    "lift_during": round(lift_during, 1),
                    "lift_after": round(lift_after, 1),
                    "before": round(before, 1),
                    "during": round(during, 1),
                    "discount": float(row.get("discount_pct", 0)),
                }
            )

        if not results:
            return _empty_fig("No data available for selected filters")

        res_df = pd.DataFrame(results).sort_values("lift_during", ascending=True)

        # Colors: positive lift = green, negative = red
        colors_during = [
            "#34A853" if v >= 0 else "#EA4335" for v in res_df["lift_during"]
        ]
        colors_after = [
            "#4285F4" if v >= 0 else "#F4B400" for v in res_df["lift_after"]
        ]

        fig = go.Figure()

        # Lift during promotion
        fig.add_trace(
            go.Bar(
                name="Lift During Promo (%)",
                y=res_df["name"],
                x=res_df["lift_during"],
                orientation="h",
                marker_color=colors_during,
                marker_opacity=0.88,
                text=[f"{v:+.1f}%" for v in res_df["lift_during"]],
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate=(
                    "<b>%{y}</b><br>Lift During: %{x:+.1f}%<br>"
                    "Baseline: %{customdata[0]:.0f}/day<br>"
                    "Discount: %{customdata[1]:.0f}%<extra></extra>"
                ),
                customdata=res_df[["before", "discount"]].values,
                xaxis="x1",
            )
        )

        # Post-promo retention dots
        fig.add_trace(
            go.Scatter(
                name="14-Day Post-Promo Lift (%)",
                y=res_df["name"],
                x=res_df["lift_after"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=8,
                    color=colors_after,
                    line=dict(width=1.5, color="#fff"),
                ),
                hovertemplate="<b>%{y}</b><br>Post-Promo Lift: %{x:+.1f}%<extra></extra>",
                xaxis="x1",
            )
        )

        # Zero reference line
        fig.add_vline(x=0, line_color="#9E9E9E", line_width=1.5, line_dash="solid")

        cat_label = f" | {category_filter}" if category_filter != "All" else ""
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=f"Promotion Demand Lift{cat_label}<br>"
                "<sup style='color:#666'>Bar = lift during promo · Dot = 14-day post-promo retention · Green = positive lift</sup>",
                font=dict(size=15, color="#4285F4"),
            ),
            xaxis=dict(
                title="Demand Lift vs Baseline (%)",
                zeroline=True,
                zerolinecolor="#BDBDBD",
                ticksuffix="%",
            ),
            yaxis=dict(tickfont=dict(size=10)),
            height=max(500, len(res_df) * 28 + 140),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
            margin=dict(t=100, b=60, l=220, r=100),
            barmode="overlay",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_store_inventory_comparison(
    region_filter: str = "All", store_id: str | None = None
) -> go.Figure:
    """
    Category Inventory Health — avg days of supply per product category.
    Correctly named: this chart shows CATEGORY-level averages, not store comparisons.
    Thresholds use per-category avg lead time (consistent with the rest of the app)
    instead of the old fixed 7d/14d values.
    """
    try:
        demand_df = get_df()
        stores = get_stores_df()
        if demand_df.empty:
            return _empty_fig("Demand data not available")

        latest_date = demand_df["date"].max()
        latest = demand_df[demand_df["date"] == latest_date].copy()
        cutoff = latest_date - pd.Timedelta(days=30)
        recent_avg = (
            demand_df[demand_df["date"] >= cutoff]
            .groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "avg_daily_demand"})
        )
        merged = latest.merge(recent_avg, on="sku_id")
        merged["days_of_supply"] = (
            merged["inventory"] / merged["avg_daily_demand"].replace(0, np.nan)
        ).fillna(0)

        # BUG-013 fix: region filter now actually works by joining transactions with
        # stores on city → region, then filtering to matching SKUs.
        txn = get_transactions()
        if (
            region_filter != "All"
            and not txn.empty
            and not stores.empty
            and "city" in txn.columns
            and "city" in stores.columns
            and "region" in stores.columns
        ):
            cities_in_region = stores[stores["region"] == region_filter][
                "city"
            ].unique()
            region_skus = txn[txn["city"].isin(cities_in_region)]["sku_id"].unique()
            if len(region_skus) > 0:
                merged = merged[merged["sku_id"].isin(region_skus)]

        if merged.empty:
            return _empty_fig(
                f"No inventory data for region: {region_filter}"
                if region_filter != "All"
                else "No inventory data available"
            )

        # Per-category avg lead time — use this as threshold instead of fixed 7/14d
        cat_lt = (
            merged.groupby("category")["lead_time_days"]
            .mean()
            .reset_index()
            .rename(columns={"lead_time_days": "avg_lt"})
        )
        cat_dos = (
            merged.groupby("category")["days_of_supply"]
            .mean()
            .reset_index()
            .sort_values("days_of_supply")
        )
        cat_dos = cat_dos.merge(cat_lt, on="category", how="left")
        cat_dos["avg_lt"] = cat_dos["avg_lt"].fillna(7)

        # Risk using per-category avg lead time — consistent with inventory dashboard
        def _cat_risk(dos, lt):
            if dos < lt:
                return "CRITICAL"
            if dos < 2 * lt:
                return "WARNING"
            return "HEALTHY"

        cat_dos["risk"] = cat_dos.apply(
            lambda r: _cat_risk(r["days_of_supply"], r["avg_lt"]), axis=1
        )
        cat_dos = cat_dos.sort_values("days_of_supply", ascending=True).reset_index(
            drop=True
        )

        color_map = {"CRITICAL": "#EA4335", "WARNING": "#F9AB00", "HEALTHY": "#34A853"}
        risk_labels = {
            "CRITICAL": "Critical (DoS < avg lead time)",
            "WARNING": "Warning (DoS < 2× avg lead time)",
            "HEALTHY": "Healthy (DoS ≥ 2× avg lead time)",
        }

        fig = go.Figure()
        for risk, color in color_map.items():
            sub = cat_dos[cat_dos["risk"] == risk]
            if sub.empty:
                continue
            fig.add_trace(
                go.Bar(
                    name=risk_labels[risk],
                    x=sub["days_of_supply"],
                    y=sub["category"],
                    orientation="h",
                    marker_color=color,
                    marker_opacity=0.88,
                    text=[f"{d:.1f}d" for d in sub["days_of_supply"]],
                    textposition="outside",
                    textfont=dict(size=10),
                    customdata=sub[["avg_lt"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Avg Days of Supply: <b>%{x:.1f}d</b><br>"
                        "Category avg lead time: %{customdata[0]:.0f}d<br>"
                        "Status: " + risk_labels[risk] + "<extra></extra>"
                    ),
                )
            )

        # Per-category lead-time markers so users can see the threshold visually
        fig.add_trace(
            go.Scatter(
                x=cat_dos["avg_lt"],
                y=cat_dos["category"],
                mode="markers",
                name="Avg lead time (critical threshold)",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color="#C5221F",
                    line=dict(color="#fff", width=1),
                ),
                hovertemplate="<b>%{y}</b><br>Avg lead time: <b>%{x:.0f}d</b><extra></extra>",
            )
        )

        n_crit = int((cat_dos["risk"] == "CRITICAL").sum())
        n_warn = int((cat_dos["risk"] == "WARNING").sum())
        scope_label = f" | Region: {region_filter}" if region_filter != "All" else ""

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    f"Category Inventory Health{scope_label}<br>"
                    "<sup style='color:#666'>Avg days of supply per category · "
                    f"{n_crit} Critical &nbsp;|&nbsp; {n_warn} Warning &nbsp;·&nbsp; "
                    "◆ = category's avg lead-time threshold</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis_title="Avg Days of Supply",
            yaxis=dict(tickfont=dict(size=11)),
            barmode="overlay",
            height=max(380, len(cat_dos) * 38 + 180),
            legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=10)),
            margin=dict(t=90, b=100, l=130, r=100),
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_stockout_risk_timeline(days_ahead: int = 30) -> go.Figure:
    """
    Stockout Risk Timeline — horizontal bar per at-risk SKU.
    Bar length = days until stockout. Colour = urgency level.
    A vertical line shows each SKU's lead time so you can see if reorder is still possible.
    """
    try:
        df = get_df()
        if df.empty:
            return _empty_fig("No inventory data available")

        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].copy()
        cutoff = latest_date - pd.Timedelta(days=30)
        recent = (
            df[df["date"] >= cutoff]
            .groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "avg_demand"})
        )
        merged = latest.merge(recent, on="sku_id")
        merged["days_until_stockout"] = (
            (merged["inventory"] / merged["avg_demand"].replace(0, np.nan))
            .fillna(999)
            .round(1)
        )
        at_risk = (
            merged[merged["days_until_stockout"] <= days_ahead]
            .sort_values("days_until_stockout")
            .reset_index(drop=True)
        )

        if at_risk.empty:
            return _empty_fig(
                f"No SKUs projected to stock out within {days_ahead} days — all well stocked!"
            )

        # Build readable label: "SKU_ID — Name"
        if "name" in at_risk.columns:
            at_risk["label"] = at_risk["sku_id"] + " — " + at_risk["name"].str[:22]
        else:
            at_risk["label"] = at_risk["sku_id"]

        def _urgency_color(d, lt):
            if d < lt:
                return "#EA4335"  # will stock out before reorder arrives
            if d <= 14:
                return "#F9AB00"  # tight but possible to reorder in time
            return "#4285F4"  # some buffer remains

        colors = [
            _urgency_color(d, lt)
            for d, lt in zip(at_risk["days_until_stockout"], at_risk["lead_time_days"])
        ]
        today = latest_date

        fig = go.Figure()

        # One trace per urgency class so legend makes sense
        urgency_groups = [
            ("#EA4335", "CRITICAL — stocks out before reorder arrives"),
            ("#F9AB00", "WARNING — reorder urgent"),
            ("#4285F4", "MONITOR — some buffer remains"),
        ]
        for col, label in urgency_groups:
            idxs = [i for i, c in enumerate(colors) if c == col]
            if not idxs:
                continue
            sub = at_risk.iloc[idxs]
            projected_dates = (
                today + sub["days_until_stockout"].apply(lambda d: pd.Timedelta(days=d))
            ).dt.strftime("%b %d")
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=sub["days_until_stockout"],
                    y=sub["label"],
                    orientation="h",
                    marker_color=col,
                    marker_opacity=0.88,
                    text=[f"{d:.0f}d" for d in sub["days_until_stockout"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    customdata=np.column_stack(
                        [
                            sub["inventory"].values,
                            sub["lead_time_days"].values,
                            projected_dates.values,
                            sub["category"].values,
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Days until stockout: <b>%{x:.0f}</b><br>"
                        "Projected stockout date: %{customdata[2]}<br>"
                        "Current stock: %{customdata[0]:,} units<br>"
                        "Lead time: %{customdata[1]:.0f}d<br>"
                        "Category: %{customdata[3]}<extra></extra>"
                    ),
                )
            )

        crit_count = sum(1 for c in colors if c == "#EA4335")
        # Note: no single avg_lt vline — each SKU has its own lead time threshold
        # which is reflected in the bar colour (CRITICAL = will stock out before
        # its own lead time elapses; not before an average lead time).
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    f"Stockout Risk — {len(at_risk)} SKUs running out within {days_ahead} days<br>"
                    f"<sup style='color:#EA4335'>{crit_count} CRITICAL (stock out before that SKU's own reorder can arrive) "
                    f"&nbsp;·&nbsp; Colour based on each SKU's individual lead time</sup>"
                ),
                font=dict(size=14, color="#EA4335"),
            ),
            xaxis_title="Days Until Stockout",
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            height=max(420, len(at_risk) * 28 + 160),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(t=90, b=60, l=200, r=100),
            barmode="overlay",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_lead_time_scatter() -> go.Figure:
    """
    Supplier Lead Time & On-Time Delivery Dashboard — horizontal bar chart.

    Replaces the illegible scatter plot (26 suppliers all at y≈5d with overlapping
    labels). New design:
    - Suppliers ranked by on-time delivery % (worst at top → easy triage)
    - Bar 1: on-time delivery % (green/amber/red by threshold)
    - Bar 2: lead-time variance (actual − promised, shown as ±d delta)
    - Target lines at 95% OTD and 0 variance
    """
    try:
        supplier_df = get_supplier_perf_df()
        products = get_products_df()
        if supplier_df.empty:
            return _empty_fig("Supplier performance data not available")

        sup_agg = (
            supplier_df.groupby("supplier_name")
            .agg(
                avg_actual_lt=("lead_time_actual_days", "mean"),
                avg_on_time=("on_time_delivery_pct", "mean"),
                avg_fill=("fill_rate_pct", "mean"),
                avg_defect=("defect_rate_pct", "mean"),
            )
            .reset_index()
        )

        # Get promised lead time from products
        if not products.empty and "supplier" in products.columns:
            promised = (
                products.groupby("supplier")["lead_time_days"]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "supplier": "supplier_name",
                        "lead_time_days": "promised_lt",
                    }
                )
            )
            sup_agg = sup_agg.merge(promised, on="supplier_name", how="left")
            sup_agg["promised_lt"] = sup_agg["promised_lt"].fillna(
                sup_agg["avg_actual_lt"]
            )
        else:
            sup_agg["promised_lt"] = sup_agg["avg_actual_lt"]

        sup_agg["lt_delta"] = (sup_agg["avg_actual_lt"] - sup_agg["promised_lt"]).round(
            1
        )
        sup_agg = sup_agg.sort_values("avg_on_time", ascending=True).reset_index(
            drop=True
        )

        # Truncate long supplier names
        sup_agg["label"] = sup_agg["supplier_name"].str[:28]

        def _otd_color(pct):
            if pct >= 95:
                return "#34A853"
            if pct >= 88:
                return "#F9AB00"
            return "#EA4335"

        otd_colors = [_otd_color(p) for p in sup_agg["avg_on_time"]]

        fig = go.Figure()

        # ── On-Time Delivery % bars ────────────────────────────────────────
        fig.add_trace(
            go.Bar(
                name="On-Time Delivery %",
                y=sup_agg["label"],
                x=sup_agg["avg_on_time"],
                orientation="h",
                marker_color=otd_colors,
                marker_opacity=0.88,
                text=[f"{v:.1f}%" for v in sup_agg["avg_on_time"]],
                textposition="outside",
                textfont=dict(size=10),
                customdata=sup_agg[
                    [
                        "avg_actual_lt",
                        "promised_lt",
                        "lt_delta",
                        "avg_fill",
                        "avg_defect",
                    ]
                ].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "On-Time Delivery: <b>%{x:.1f}%</b><br>"
                    "Actual Lead Time: %{customdata[0]:.1f}d &nbsp;|&nbsp; "
                    "Promised: %{customdata[1]:.0f}d<br>"
                    "Lead Time Delta: <b>%{customdata[2]:+.1f}d</b><br>"
                    "Fill Rate: %{customdata[3]:.1f}% &nbsp;|&nbsp; "
                    "Defect Rate: %{customdata[4]:.2f}%"
                    "<extra></extra>"
                ),
                xaxis="x",
            )
        )

        # ── Lead-time delta bars (secondary x-axis) ────────────────────────
        delta_colors = ["#EA4335" if d > 0 else "#34A853" for d in sup_agg["lt_delta"]]
        fig.add_trace(
            go.Bar(
                name="Lead Time Delta (actual − promised)",
                y=sup_agg["label"],
                x=sup_agg["lt_delta"],
                orientation="h",
                marker_color=delta_colors,
                marker_opacity=0.6,
                text=[f"{d:+.1f}d" for d in sup_agg["lt_delta"]],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Lead Time Delta: <b>%{x:+.1f}d</b><br>"
                    "(positive = delivered late, negative = delivered early)"
                    "<extra></extra>"
                ),
                xaxis="x2",
            )
        )

        n_poor = int((sup_agg["avg_on_time"] < 88).sum())
        n_ok = int((sup_agg["avg_on_time"] >= 95).sum())

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    "Supplier Lead Time & On-Time Delivery<br>"
                    f"<sup style='color:#666'>{n_ok} suppliers meeting 95% target &nbsp;|&nbsp; "
                    f"{n_poor} underperforming (&lt;88%) — sorted worst → best</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis=dict(
                title="On-Time Delivery %",
                range=[75, 105],
                domain=[0, 0.62],
                ticksuffix="%",
            ),
            xaxis2=dict(
                title="Lead Time Delta (days)",
                range=[
                    min(-3, sup_agg["lt_delta"].min() * 1.3),
                    max(3, sup_agg["lt_delta"].max() * 1.3),
                ],
                domain=[0.68, 1.0],
                zeroline=True,
                zerolinecolor="#6B7280",
                zerolinewidth=1.5,
            ),
            yaxis=dict(tickfont=dict(size=10)),
            barmode="overlay",
            height=max(480, len(sup_agg) * 26 + 160),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
            margin=dict(t=90, b=60, l=210, r=60),
        )

        # Target line at 95% OTD
        fig.add_vline(
            x=95,
            line_dash="dash",
            line_color="#34A853",
            line_width=1.5,
            annotation_text="Target (95%)",
            annotation_font=dict(color="#137333", size=10),
            annotation_position="top",
        )

        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_cold_chain_trend(sku_filter: str | None = None) -> go.Figure:
    """
    Cold Chain Monitor — two-row subplot layout.

    Row 1: Weekly-average temperature per SKU (smooth lines).
           730 raw daily points → ~104 weekly points = readable.
           Reference lines at 0°C and 6°C mark the safe zone boundaries.

    Row 2: Monthly breach count per SKU (grouped bars).
           A breach = any day where temp_celsius < 0 or > 6.

    Replaces the raw daily line chart where 3 SKUs × 730 days
    produced an unreadable noise band.
    """
    try:
        cc = get_cold_chain_df()
        if cc.empty:
            return _empty_fig("Cold chain data not available")

        if sku_filter and sku_filter != "All":
            cc = cc[cc["sku_id"] == sku_filter]
            if cc.empty:
                return _empty_fig(f"No cold chain data for SKU: {sku_filter}")

        cc = cc.copy()
        cc["date"] = pd.to_datetime(cc["date"])

        # ── Week-level aggregation ─────────────────────────────────────────
        cc["week"] = cc["date"].dt.to_period("W").dt.start_time
        weekly = (
            cc.groupby(["sku_id", "week"])
            .agg(
                avg_temp=("temp_celsius", "mean"),
                min_temp=("temp_celsius", "min"),
                max_temp=("temp_celsius", "max"),
                breach_days=(
                    "temp_breach",
                    lambda x: (x.astype(str).str.lower() == "true").sum(),
                ),
                name=("name", "first"),
            )
            .reset_index()
        )

        # ── Monthly breach count ───────────────────────────────────────────
        cc["month"] = cc["date"].dt.to_period("M").dt.start_time
        monthly_breach = (
            cc[cc["temp_breach"].astype(str).str.lower() == "true"]
            .groupby(["sku_id", "month"])
            .size()
            .reset_index(name="breach_count")
        )
        # Add name
        sku_names = cc.drop_duplicates("sku_id")[["sku_id", "name"]].set_index(
            "sku_id"
        )["name"]
        monthly_breach["name"] = monthly_breach["sku_id"].map(sku_names).str[:25]

        skus = cc["sku_id"].unique()
        palette = ["#4285F4", "#EA4335", "#34A853", "#F9AB00", "#7C3AED"]

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.65, 0.35],
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Weekly Average Temperature per SKU",
                "Monthly Breach Count",
            ),
        )

        # ── Row 1: weekly temperature lines ───────────────────────────────
        for i, sku in enumerate(skus):
            sub = weekly[weekly["sku_id"] == sku].sort_values("week")
            name_label = sub["name"].iloc[0][:25] if not sub.empty else sku
            color = palette[i % len(palette)]
            # Shade the uncertainty band (min–max)
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([sub["week"], sub["week"].iloc[::-1]]),
                    y=pd.concat([sub["max_temp"], sub["min_temp"].iloc[::-1]]),
                    fill="toself",
                    # BUG-013 fix: proper hex→rgba conversion
                    fillcolor=(
                        lambda h, a=0.10: (
                            f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"
                        )
                    )(color)
                    if color.startswith("#")
                    else color,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{name_label} range",
                ),
                row=1,
                col=1,
            )
            # Weekly avg line
            fig.add_trace(
                go.Scatter(
                    x=sub["week"],
                    y=sub["avg_temp"],
                    mode="lines+markers",
                    name=name_label,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    customdata=sub[["min_temp", "max_temp", "breach_days"]].values,
                    hovertemplate=(
                        f"<b>{name_label}</b><br>"
                        "%{x|%b %d %Y}<br>"
                        "Avg: <b>%{y:.1f}°C</b><br>"
                        "Range: %{customdata[0]:.1f}°C – %{customdata[1]:.1f}°C<br>"
                        "Breach days this week: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        # Safe zone reference lines (cleaner than filled rects)
        for y_val, label, color in [
            (0, "Min safe (0°C)", "#4285F4"),
            (6, "Max safe (6°C)", "#EA4335"),
        ]:
            fig.add_hline(
                y=y_val,
                line_dash="dash",
                line_color=color,
                line_width=1.2,
                annotation_text=label,
                annotation_font=dict(size=9, color=color),
                annotation_position="left",
                row=1,
                col=1,
            )

        # ── Row 2: monthly breach bars ─────────────────────────────────────
        for i, sku in enumerate(skus):
            sub = monthly_breach[monthly_breach["sku_id"] == sku].sort_values("month")
            if sub.empty:
                continue
            name_label = sub["name"].iloc[0] if not sub.empty else sku
            fig.add_trace(
                go.Bar(
                    x=sub["month"],
                    y=sub["breach_count"],
                    name=f"{name_label} breaches",
                    marker_color=palette[i % len(palette)],
                    marker_opacity=0.75,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{name_label}</b><br>"
                        "%{x|%b %Y}: <b>%{y} breach days</b><extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

        total_breaches = int(
            (cc["temp_breach"].astype(str).str.lower() == "true").sum()
        )
        scope = sku_names.get(sku_filter, sku_filter) if sku_filter else "All SKUs"

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    f"Cold Chain Monitor — {scope}<br>"
                    f"<sup style='color:#666'>{total_breaches} total breach days across full period &nbsp;·&nbsp; "
                    f"Safe zone: 0°C – 6°C</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            height=600,
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(t=100, b=60, l=60, r=40),
            hovermode="x unified",
            barmode="group",
        )
        fig.update_yaxes(title_text="Temp (°C)", range=[-3, 10], row=1, col=1)
        fig.update_yaxes(title_text="Breach Days", row=2, col=1)
        fig.update_xaxes(
            title_text="Month",
            tickformat="%b %Y",
            tickangle=30,
            tickfont=dict(size=10),
            row=2,
            col=1,
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_reorder_events_timeline() -> go.Figure:
    """
    Reorder Events Timeline — bar chart by month, colored by category.
    Reorder events = days where inventory increased vs. previous day.
    """
    try:
        df = get_df()
        if df.empty:
            return _empty_fig("No inventory data available")

        df_sorted = df.sort_values(["sku_id", "date"])
        df_sorted["inv_delta"] = df_sorted.groupby("sku_id")["inventory"].diff()
        # BUG-033 fix: only count significant inventory increases (≥ avg lead-time
        # demand) as reorder events, filtering out small corrections/adjustments.
        avg_daily = df_sorted.groupby("sku_id")["demand"].transform("mean")
        lead_time_days = df_sorted["lead_time_days"]
        min_reorder_qty = (avg_daily * lead_time_days * 0.25).clip(lower=1)
        reorders = df_sorted[df_sorted["inv_delta"] >= min_reorder_qty].copy()
        reorders["month"] = reorders["date"].dt.to_period("M").dt.start_time
        monthly = (
            reorders.groupby(["month", "category"])
            .size()
            .reset_index(name="reorder_events")
        )

        categories = sorted(monthly["category"].unique().tolist())
        # Distinct, high-contrast palette
        palette = [
            "#4285F4",
            "#EA4335",
            "#34A853",
            "#F9AB00",
            "#7C3AED",
            "#0891B2",
            "#F97316",
            "#EC4899",
            "#14B8A6",
            "#84CC16",
        ]
        cat_colors = {
            cat: palette[i % len(palette)] for i, cat in enumerate(categories)
        }

        fig = go.Figure()
        for cat in categories:
            sub = monthly[monthly["category"] == cat].sort_values("month")
            fig.add_trace(
                go.Bar(
                    x=sub["month"],
                    y=sub["reorder_events"],
                    name=cat,
                    marker_color=cat_colors.get(cat, "#9CA3AF"),
                    marker_opacity=0.85,
                    hovertemplate=f"<b>{cat}</b><br>%{{x|%b %Y}}: %{{y}} reorder events<extra></extra>",
                )
            )

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    "Reorder Events by Month & Category<br>"
                    "<sup style='color:#666'>Each bar segment = purchase orders placed for that category that month</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis=dict(
                title="Month",
                tickformat="%b %Y",
                tickangle=35,
                tickfont=dict(size=10),
            ),
            yaxis_title="Number of Reorder Events",
            barmode="stack",
            height=480,
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
            margin=dict(t=90, b=80),
            hovermode="x unified",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_dead_stock_bar(category_filter: str = "All") -> go.Figure:
    """
    Dead Stock Analysis — horizontal bar per SKU coloured by classification.
    Dead = 0 demand in last 60 days (red).
    Slow = avg 60-day demand < 50% of overall avg (amber).
    Active = everything else (green).
    Shows each SKU individually so dead/slow movers are immediately identifiable.
    """
    try:
        df = get_df()
        products = get_products_df()
        if df.empty:
            return _empty_fig("No inventory data available")

        latest_date = df["date"].max()
        cutoff60 = latest_date - pd.Timedelta(days=60)

        # Demand in last 60 days per SKU
        recent = (
            df[df["date"] >= cutoff60]
            .groupby("sku_id")["demand"]
            .agg(["sum", "mean"])
            .reset_index()
        )
        recent.columns = ["sku_id", "demand_60d", "avg_demand_60d"]

        # Overall historical avg demand per SKU
        overall_avg = (
            df.groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "overall_avg_demand"})
        )

        # BUG-017 fix: also compute per-category average so "Slow Moving" threshold
        # is relative to the category norm, not a single global number.
        cat_avg = (
            df.groupby(["sku_id", "category"])["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "cat_avg_demand"})
        )
        # Use category-level avg per SKU as the slow-moving reference
        # An SKU is slow-moving if its 60d avg < 50% of its category peer average
        cat_peer_avg = (
            df.groupby("category")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "cat_peer_avg"})
        )

        # Latest inventory snapshot
        latest_inv = df[df["date"] == latest_date][
            ["sku_id", "inventory", "category", "name"]
        ].copy()

        merged = (
            latest_inv.merge(recent, on="sku_id", how="left")
            .merge(overall_avg, on="sku_id", how="left")
            .merge(cat_peer_avg, on="category", how="left")
        )

        # Add cost info for value calculation
        if not products.empty and "cost_inr" in products.columns:
            merged = merged.merge(
                products[["sku_id", "cost_inr"]], on="sku_id", how="left"
            )
            merged["cost_inr"] = merged["cost_inr"].fillna(0)
        else:
            merged["cost_inr"] = 0

        merged["inventory_value"] = merged["inventory"] * merged["cost_inr"]
        merged["demand_60d"] = merged["demand_60d"].fillna(0)
        merged["avg_demand_60d"] = merged["avg_demand_60d"].fillna(0)
        merged["overall_avg_demand"] = merged["overall_avg_demand"].fillna(0)
        merged["cat_peer_avg"] = merged["cat_peer_avg"].fillna(
            merged["overall_avg_demand"]
        )

        def _classify(row):
            if row["demand_60d"] == 0:
                return "Dead Stock"
            # Slow Moving: recent avg < 50% of category peer average (BUG-017 fix)
            if row["avg_demand_60d"] < row["cat_peer_avg"] * 0.5:
                return "Slow Moving"
            return "Active"

        merged["stock_class"] = merged.apply(_classify, axis=1)

        # Apply category filter
        if category_filter != "All":
            merged = merged[merged["category"] == category_filter]

        if merged.empty:
            return _empty_fig(f"No data for: {category_filter}")

        # Sort: Dead first, then Slow, then Active; within each class by value desc
        class_order = {"Dead Stock": 0, "Slow Moving": 1, "Active": 2}
        merged["_sort_key"] = merged["stock_class"].map(class_order)
        merged = merged.sort_values(
            ["_sort_key", "inventory_value"], ascending=[True, False]
        ).reset_index(drop=True)

        # Build label: "SKU_ID — Name (Category)"
        if "name" in merged.columns:
            merged["label"] = merged["sku_id"] + " — " + merged["name"].str[:22]
        else:
            merged["label"] = merged["sku_id"]

        color_map = {
            "Dead Stock": "#EA4335",
            "Slow Moving": "#F9AB00",
            "Active": "#34A853",
        }
        colors = merged["stock_class"].map(color_map).fillna("#9CA3AF")

        # Hover text
        hover = [
            (
                f"<b>{row.sku_id}</b><br>"
                f"Category: {row.category}<br>"
                f"Status: <b>{row.stock_class}</b><br>"
                f"Inventory: {int(row.inventory):,} units<br>"
                f"Inventory Value: ₹{row.inventory_value:,.0f}<br>"
                f"Demand last 60d: {int(row.demand_60d):,} units<br>"
                f"Avg daily demand (60d): {row.avg_demand_60d:.1f}<br>"
                f"Overall avg daily demand: {row.overall_avg_demand:.1f}"
            )
            for _, row in merged.iterrows()
        ]

        fig = go.Figure()

        # One trace per class so legend entries are correct
        for cls in ["Dead Stock", "Slow Moving", "Active"]:
            mask = merged["stock_class"] == cls
            sub = merged[mask]
            if sub.empty:
                continue
            fig.add_trace(
                go.Bar(
                    name=cls,
                    y=sub["label"],
                    x=sub["inventory_value"],
                    orientation="h",
                    marker_color=color_map[cls],
                    marker_opacity=0.88,
                    customdata=sub[
                        ["sku_id", "category", "inventory", "demand_60d"]
                    ].values,
                    text=[f"₹{v:,.0f}" for v in sub["inventory_value"]],
                    textposition="outside",
                    textfont=dict(size=9),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Category: %{customdata[1]}<br>"
                        f"Status: <b>{cls}</b><br>"
                        "Stock: %{customdata[2]:,} units<br>"
                        "Demand (60d): %{customdata[3]:,}<br>"
                        "Value: ₹%{x:,.0f}<extra></extra>"
                    ),
                )
            )

        n_skus = len(merged)
        scope = category_filter if category_filter != "All" else "All Categories"
        dead_count = int((merged["stock_class"] == "Dead Stock").sum())
        slow_count = int((merged["stock_class"] == "Slow Moving").sum())

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    f"Dead Stock Analysis by SKU — {scope}<br>"
                    f"<sup style='color:#666'>{dead_count} Dead &nbsp;|&nbsp; "
                    f"{slow_count} Slow Moving &nbsp;|&nbsp; "
                    f"{n_skus - dead_count - slow_count} Active &nbsp;(of {n_skus} SKUs)</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis_title="Inventory Value (₹)",
            yaxis=dict(
                tickfont=dict(size=10),
                autorange="reversed",
            ),
            barmode="overlay",
            height=max(480, n_skus * 22 + 180),
            legend=dict(
                orientation="h",
                y=-0.07,  # below the chart — never overlaps bars
                x=0,
                font=dict(size=11),
                yanchor="top",
            ),
            margin=dict(t=80, b=90, l=190, r=100),
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_seasonal_demand_radar() -> go.Figure:
    """
    Seasonal Demand Index — grouped bar chart (one group per month, one bar per category).
    Index = (month demand / annual monthly avg) × 100.  100 = average month.
    Far more readable than a radar: magnitudes are directly comparable bar-by-bar.
    """
    try:
        df = get_df()
        if df.empty:
            return _empty_fig("No demand data available")

        df["month_num"] = df["date"].dt.month
        # BUG-015 fix: use mean not sum so months with different day counts
        # (Feb=28, Mar=31) are not penalised for having fewer rows.
        monthly = df.groupby(["category", "month_num"])["demand"].mean().reset_index()
        annual_avg = (
            df.groupby(["category", "month_num"])["demand"]
            .mean()
            .groupby(level="category")
            .mean()
            .reset_index()
            .rename(columns={"demand": "annual_monthly_avg"})
        )
        monthly = monthly.merge(annual_avg, on="category")
        monthly["index"] = (
            monthly["demand"] / monthly["annual_monthly_avg"].replace(0, np.nan) * 100
        ).round(1)

        # Top 6 categories by total mean demand
        # BUG-1 fix: 'annual' was never defined — compute it from annual_avg
        annual = (
            df.groupby("category")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "annual"})
        )
        top6 = annual.nlargest(6, "annual")["category"].tolist()
        monthly = monthly[monthly["category"].isin(top6)]

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        palette = [
            "#4285F4",
            "#EA4335",
            "#34A853",
            "#F9AB00",
            "#7C3AED",
            "#0891B2",
        ]

        fig = go.Figure()
        for i, cat in enumerate(top6):
            sub = monthly[monthly["category"] == cat].sort_values("month_num")
            vals = (
                sub.set_index("month_num")["index"]
                .reindex(range(1, 13), fill_value=100)
                .tolist()
            )
            # Colour bars: peak months brighter, below-average muted
            # bar_colors: brighter for above-average months, muted for below-average
            # Applied per-bar so peak months are visually prominent
            base_color = palette[i % len(palette)]
            bar_colors = [
                base_color
                if v >= 100
                else base_color + "66"  # 40% opacity for below-average months
                for v in vals
            ]
            fig.add_trace(
                go.Bar(
                    name=cat,
                    x=month_names,
                    y=vals,
                    marker_color=bar_colors,  # now correctly applied per-bar
                    marker_opacity=1.0,  # opacity handled via hex alpha above
                    hovertemplate=(
                        f"<b>{cat}</b><br>"
                        "Month: %{x}<br>"
                        "Demand Index: %{y:.0f}<br>"
                        "<i>(100 = average month)</i><extra></extra>"
                    ),
                )
            )

        # Reference line at 100 = average
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="#9CA3AF",
            line_width=1.5,
            annotation_text="Average (100)",
            annotation_font=dict(size=10, color="#6B7280"),
            annotation_position="bottom right",
        )

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    "Seasonal Demand Index by Category<br>"
                    "<sup style='color:#666'>Index 100 = average month. Above 100 = peak season, below = slow.</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis_title="Month",
            yaxis_title="Demand Index (avg month = 100)",
            barmode="group",
            height=500,
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
            margin=dict(t=90, b=70),
            hovermode="x unified",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_financial_kpi_cards() -> str:
    """
    Financial KPI Summary — returns HTML string with KPI cards.
    """
    try:
        df = get_df()
        products = get_products_df()
        if df.empty:
            return "<div style='padding:20px;color:#9CA3AF;'>Inventory data not available</div>"

        latest_date = df["date"].max()
        latest_inv = df[df["date"] == latest_date][["sku_id", "inventory"]].copy()

        if (
            not products.empty
            and "cost_inr" in products.columns
            and "price_inr" in products.columns
        ):
            prod_slim = products[["sku_id", "cost_inr", "price_inr"]].drop_duplicates(
                "sku_id"
            )
            merged = latest_inv.merge(prod_slim, on="sku_id", how="left").fillna(
                {"cost_inr": 0, "price_inr": 0}
            )
        else:
            # Use daily demand CSV columns if available
            latest_full = df[df["date"] == latest_date].copy()
            if "cost_inr" in latest_full.columns and "price_inr" in latest_full.columns:
                merged = latest_full[
                    ["sku_id", "inventory", "cost_inr", "price_inr"]
                ].copy()
            else:
                merged = latest_inv.copy()
                merged["cost_inr"] = 0
                merged["price_inr"] = 0

        total_inv_val = (merged["inventory"] * merged["cost_inr"]).sum()
        potential_retail = (merged["inventory"] * merged["price_inr"]).sum()

        # Dead stock (0 demand in last 60 days)
        cutoff60 = latest_date - pd.Timedelta(days=60)
        demand_60d = (
            df[df["date"] >= cutoff60].groupby("sku_id")["demand"].sum().reset_index()
        )
        demand_60d.columns = ["sku_id", "demand_60d"]
        merged2 = merged.merge(demand_60d, on="sku_id", how="left").fillna(
            {"demand_60d": 0}
        )
        dead_value = (
            merged2[merged2["demand_60d"] == 0]["inventory"]
            * merged2[merged2["demand_60d"] == 0]["cost_inr"]
        ).sum()

        # BUG-004 fix: only count stockout days in the last 30 days, not all-time
        cutoff30_kpi = latest_date - pd.Timedelta(days=30)
        recent30 = df[df["date"] >= cutoff30_kpi].copy()
        zero_inv_days = (
            recent30[recent30["inventory"] == 0]
            .groupby("sku_id")
            .size()
            .reset_index(name="zero_days")
        )
        avg_demand = (
            recent30.groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "avg_demand"})
        )
        stockout = (
            zero_inv_days.merge(avg_demand, on="sku_id")
            .merge(merged[["sku_id", "price_inr"]], on="sku_id", how="left")
            .fillna({"price_inr": 0})
        )
        lost_revenue = (
            stockout["zero_days"] * stockout["avg_demand"] * stockout["price_inr"]
        ).sum()

        # Working capital days
        daily_df = df.copy()
        if "cost_inr" in daily_df.columns and "price_inr" in daily_df.columns:
            daily_df["inv_value"] = daily_df["inventory"] * daily_df["cost_inr"]
            daily_df["daily_revenue"] = daily_df["demand"] * daily_df["price_inr"]
        else:
            daily_df["inv_value"] = daily_df["inventory"]
            daily_df["daily_revenue"] = daily_df["demand"]
        avg_inv_val = daily_df.groupby("date")["inv_value"].sum().mean()
        avg_daily_rev = daily_df.groupby("date")["daily_revenue"].sum().mean()
        wc_days = avg_inv_val / avg_daily_rev if avg_daily_rev > 0 else 0

        cards = [
            (
                "#4285F4",
                "#E8F0FE",
                "Total Inventory Value",
                _fmt_inr(total_inv_val),
                "At cost",
            ),
            (
                "#34A853",
                "#E6F4EA",
                "Potential Retail Value",
                _fmt_inr(potential_retail),
                "At MRP",
            ),
            (
                "#EA4335",
                "#FCE8E6",
                "Dead Stock Value",
                _fmt_inr(dead_value),
                "0 demand 60d",
            ),
            (
                "#FBBC05",
                "#FEF9E0",
                "Est. Lost Revenue",
                _fmt_inr(lost_revenue),
                "Stockout days (last 30d)",
            ),
            (
                "#7C3AED",
                "#F3E8FF",
                "Days Inventory Outstanding",
                f"{wc_days:.1f}d",
                "Inventory value ÷ daily revenue",
            ),
        ]
        inner = ""
        for border_c, bg_c, label, val, sub in cards:
            inner += f"""
            <div style="flex:1;min-width:160px;background:{bg_c};border:2px solid {border_c};
                        border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:1.65rem;font-weight:800;color:{border_c};">{val}</div>
                <div style="font-size:0.85rem;font-weight:700;color:{border_c};margin-top:4px;">{label}</div>
                <div style="font-size:0.75rem;color:#5F6368;margin-top:2px;">{sub}</div>
            </div>"""
        return f'<div style="display:flex;gap:12px;margin:14px 0;flex-wrap:wrap;">{inner}</div>'
    except Exception as exc:
        return f"<div style='color:#EA4335;padding:12px;'>Error computing KPIs: {exc}</div>"


def build_private_label_vs_third_party(store_id: str | None = None) -> go.Figure:
    """
    Private Label vs Third Party Revenue — stacked horizontal bar per category.
    Shows both absolute split and % share for each product category.
    """
    try:
        txn = get_transactions()
        if store_id:
            txn = txn[txn["store_id"] == store_id]
        products = get_products_df()
        if txn.empty or products.empty:
            return _empty_fig("Transaction or product data not available")

        prod_slim = products[["sku_id", "brand_type"]].drop_duplicates("sku_id")
        merged = txn.merge(prod_slim, on="sku_id", how="left")
        merged["brand_type"] = merged["brand_type"].fillna("Third Party")

        # Revenue per category × brand_type
        agg = (
            merged.groupby(["category", "brand_type"])["net_revenue_inr"]
            .sum()
            .reset_index()
        )
        pivot = agg.pivot(
            index="category", columns="brand_type", values="net_revenue_inr"
        ).fillna(0)
        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("total", ascending=True)

        color_map = {"Private Label": "#4285F4", "Third Party": "#EA4335"}
        fig = go.Figure()

        for btype in ["Third Party", "Private Label"]:
            if btype not in pivot.columns:
                continue
            pct = (pivot[btype] / pivot["total"] * 100).round(1)
            fig.add_trace(
                go.Bar(
                    name=btype,
                    y=pivot.index,
                    x=pivot[btype],
                    orientation="h",
                    marker_color=color_map.get(btype, "#9CA3AF"),
                    marker_opacity=0.88,
                    text=[f"{p:.0f}%" for p in pct],
                    textposition="inside",
                    textfont=dict(size=10, color="#fff"),
                    hovertemplate=(
                        f"<b>%{{y}}</b> — {btype}<br>"
                        "Revenue: ₹%{x:,.0f}<br>"
                        "Share: %{text}<extra></extra>"
                    ),
                )
            )

        total_rev = pivot["total"].sum()
        pl_rev = pivot.get("Private Label", pd.Series([0])).sum()
        pl_pct = pl_rev / total_rev * 100 if total_rev > 0 else 0

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    "Private Label vs Third Party Revenue by Category<br>"
                    f"<sup style='color:#666'>Private Label: {_fmt_inr(pl_rev)} ({pl_pct:.1f}% of total revenue)</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis_title="Revenue (₹)",
            yaxis=dict(tickfont=dict(size=11)),
            barmode="stack",
            height=max(420, len(pivot) * 38 + 140),
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
            margin=dict(t=90, b=60, l=120, r=60),
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_mom_growth_chart(months: int = 12, store_id: str | None = None) -> go.Figure:
    """
    Month-over-Month Growth Chart — revenue, units, margin indexed to 100.
    """
    try:
        txn = get_transactions()
        if store_id:
            txn = txn[txn["store_id"] == store_id]
        if txn.empty:
            return _empty_fig("Sales transaction data not available")

        # Guard: verify required columns exist before aggregating (H-07 fix)
        required = {"date", "net_revenue_inr", "quantity", "gross_margin_inr"}
        missing = required - set(txn.columns)
        if missing:
            return _empty_fig(
                f"Transactions CSV missing columns: {', '.join(sorted(missing))}"
            )

        txn["month"] = txn["date"].dt.to_period("M")
        monthly = (
            txn.groupby("month")
            .agg(
                revenue=("net_revenue_inr", "sum"),
                units=("quantity", "sum"),
                margin=("gross_margin_inr", "sum"),
            )
            .reset_index()
            .sort_values("month")
        )

        monthly = monthly.tail(months).reset_index(drop=True)
        if monthly.empty:
            return _empty_fig("Not enough monthly data")

        # Index to 100 at first non-zero month to avoid division-by-zero (BUG-027)
        base_rev = (
            float(monthly["revenue"].iloc[0]) or float(monthly["revenue"].max()) or 1.0
        )
        base_units = (
            float(monthly["units"].iloc[0]) or float(monthly["units"].max()) or 1.0
        )
        base_margin = (
            float(monthly["margin"].iloc[0]) or float(monthly["margin"].max()) or 1.0
        )
        monthly["rev_idx"] = monthly["revenue"] / base_rev * 100
        monthly["units_idx"] = monthly["units"] / base_units * 100
        monthly["margin_idx"] = monthly["margin"] / base_margin * 100
        monthly["month_str"] = monthly["month"].astype(str)
        monthly["rev_growth_pct"] = monthly["revenue"].pct_change() * 100

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["month_str"],
                y=monthly["rev_idx"],
                mode="lines+markers",
                name="Revenue",
                line=dict(color="#4285F4", width=2.5),
                marker=dict(size=7, color="#4285F4"),
                customdata=monthly["revenue"].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Revenue Index: %{y:.1f}<br>"
                    "Actual Revenue: ₹%{customdata:,.0f}<extra></extra>"
                ),
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month_str"],
                y=monthly["units_idx"],
                mode="lines+markers",
                name="Units Sold",
                line=dict(color="#34A853", width=2, dash="dot"),
                marker=dict(size=6, color="#34A853"),
                customdata=monthly["units"].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Units Index: %{y:.1f}<br>"
                    "Actual Units: %{customdata:,}<extra></extra>"
                ),
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month_str"],
                y=monthly["margin_idx"],
                mode="lines+markers",
                name="Gross Margin",
                line=dict(color="#EA4335", width=2, dash="dash"),
                marker=dict(size=6, color="#EA4335"),
                customdata=monthly["margin"].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Margin Index: %{y:.1f}<br>"
                    "Actual Margin: ₹%{customdata:,.0f}<extra></extra>"
                ),
                yaxis="y1",
            )
        )

        # Reference line at base=100
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="#9CA3AF",
            line_width=1,
            annotation_text="Base (100)",
            annotation_font=dict(size=9, color="#6B7280"),
            annotation_position="bottom right",
        )

        # Annotate MoM % changes on revenue line (only non-zero changes)
        for i, row in monthly.iterrows():
            if not pd.isna(row["rev_growth_pct"]) and abs(row["rev_growth_pct"]) > 0.5:
                color = "#34A853" if row["rev_growth_pct"] >= 0 else "#EA4335"
                fig.add_annotation(
                    x=row["month_str"],
                    y=row["rev_idx"],
                    text=f"{row['rev_growth_pct']:+.1f}%",
                    showarrow=False,
                    font=dict(size=8, color=color),
                    yshift=14,
                )

        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=(
                    "Month-over-Month Growth Trend<br>"
                    "<sup style='color:#666'>Index = 100 at first month. Hover for actual values. % labels = MoM revenue change.</sup>"
                ),
                font=dict(size=14, color="#4285F4"),
            ),
            xaxis=dict(title="Month", tickangle=40, tickfont=dict(size=10)),
            yaxis=dict(title="Growth Index (first month = 100)"),
            height=480,
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
            margin=dict(t=90, b=80),
            hovermode="x unified",
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_customer_segment_donut(
    category_filter: str = "All",
    channel_filter: str = "All",
    period_days: int = 365,
    store_id: str | None = None,
) -> go.Figure:
    """
    Customer Segment Revenue — stacked bar chart (more readable than donut).
    Shows revenue, units, and avg order value per segment.
    Filters: category, channel, period, store_id.
    """
    try:
        txn = get_transactions()
        if store_id:
            txn = txn[txn["store_id"] == store_id]
        if txn.empty:
            return _empty_fig("Sales transaction data not available")

        if "customer_segment" not in txn.columns:
            return _empty_fig("customer_segment column not found in transactions")

        cutoff = txn["date"].max() - pd.Timedelta(days=period_days)
        txn = txn[txn["date"] >= cutoff].copy()
        if category_filter != "All":
            txn = txn[txn["category"] == category_filter]
        if channel_filter != "All":
            txn = txn[txn["channel"] == channel_filter]

        if txn.empty:
            return _empty_fig("No data for selected filters")

        agg = (
            txn.groupby("customer_segment")
            .agg(
                revenue=("net_revenue_inr", "sum"),
                units=("quantity", "sum"),
                orders=("txn_id", "count"),
                margin=("gross_margin_inr", "sum"),
            )
            .reset_index()
            .sort_values("revenue", ascending=False)
        )
        agg["avg_order_value"] = agg["revenue"] / agg["orders"].replace(0, 1)
        agg["margin_pct"] = (
            agg["margin"] / agg["revenue"].replace(0, np.nan) * 100
        ).fillna(0)
        total = agg["revenue"].sum()

        seg_colors = [
            "#4285F4",
            "#34A853",
            "#EA4335",
            "#FBBC05",
            "#7C3AED",
            "#0891B2",
            "#F97316",
        ]

        fig = go.Figure()

        # Revenue bars
        fig.add_trace(
            go.Bar(
                name="Revenue (₹)",
                x=agg["customer_segment"],
                y=agg["revenue"],
                marker_color=seg_colors[: len(agg)],
                marker_opacity=0.85,
                text=[
                    f"{_fmt_inr(r)}<br>{p:.0f}% margin"
                    for r, p in zip(agg["revenue"], agg["margin_pct"])
                ],
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate=(
                    "<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<br>"
                    "Orders: %{customdata[0]:,}<br>"
                    "Avg Order: ₹%{customdata[1]:,.0f}<extra></extra>"
                ),
                customdata=agg[["orders", "avg_order_value"]].values,
            )
        )

        # Avg order value line on secondary axis
        fig.add_trace(
            go.Scatter(
                name="Avg Order Value (₹)",
                x=agg["customer_segment"],
                y=agg["avg_order_value"],
                mode="lines+markers",
                marker=dict(
                    size=9, color="#EA4335", line=dict(width=2, color="#C5221F")
                ),
                line=dict(color="#EA4335", width=2, dash="dot"),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Avg Order: ₹%{y:,.0f}<extra></extra>",
            )
        )

        subtitle = (
            (f"{category_filter}" if category_filter != "All" else "All Categories")
            + (f" | {channel_filter}" if channel_filter != "All" else "")
            + f" | Last {period_days} days"
        )
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=f"Revenue by Customer Segment — {subtitle}",
                font=dict(size=15, color="#4285F4"),
            ),
            xaxis=dict(title="Customer Segment", tickfont=dict(size=11)),
            yaxis=dict(title="Total Revenue (₹)", side="left"),
            yaxis2=dict(
                title="Avg Order Value (₹)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            height=480,
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
            margin=dict(t=90, b=80, r=80),
            bargap=0.25,
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


def build_top_skus_bar(
    category_filter: str = "All",
    channel_filter: str = "All",
    top_n: int = 10,
    store_id: str | None = None,
) -> go.Figure:
    """
    Top N SKUs by Revenue — horizontal bar chart.
    """
    try:
        txn = get_transactions()
        if store_id:
            txn = txn[txn["store_id"] == store_id]
        if txn.empty:
            return _empty_fig("Sales transaction data not available")

        filtered = txn.copy()
        if category_filter != "All":
            filtered = filtered[filtered["category"] == category_filter]
        if channel_filter != "All":
            filtered = filtered[filtered["channel"] == channel_filter]

        sku_rev = (
            filtered.groupby(["sku_id", "category"])["net_revenue_inr"]
            .sum()
            .reset_index()
            .sort_values("net_revenue_inr", ascending=False)
            .head(top_n)
        )

        if sku_rev.empty:
            return _empty_fig(
                f"No data for filters: {category_filter} / {channel_filter}"
            )

        sku_rev = sku_rev.sort_values("net_revenue_inr", ascending=True)
        cats = sku_rev["category"].unique().tolist()
        cat_colors = dict(zip(sorted(cats), px.colors.qualitative.Set2))
        bar_colors = [cat_colors.get(c, "#9CA3AF") for c in sku_rev["category"]]

        fig = go.Figure(
            go.Bar(
                x=sku_rev["net_revenue_inr"],
                y=sku_rev["sku_id"],
                orientation="h",
                marker_color=bar_colors,
                text=sku_rev["net_revenue_inr"].apply(_fmt_inr),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>",
            )
        )
        title_parts = [f"Top {top_n} SKUs by Revenue"]
        if category_filter != "All":
            title_parts.append(category_filter)
        if channel_filter != "All":
            title_parts.append(channel_filter)
        fig.update_layout(
            **_CHART_LAYOUT,
            title=dict(
                text=" — ".join(title_parts), font=dict(size=15, color="#4285F4")
            ),
            xaxis_title="Net Revenue (₹)",
            height=max(380, top_n * 36 + 120),
            margin=dict(t=80, b=60, l=120, r=120),
        )
        return fig
    except Exception as exc:
        return _empty_fig(f"Error: {exc}")


# Tab 2: Inventory Dashboard


def build_inventory_fig(category_filter: str):
    df = get_df()
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    cutoff = latest_date - pd.Timedelta(days=30)
    recent = (
        df[df["date"] >= cutoff]
        .groupby("sku_id")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "avg_daily_demand"})
    )
    merged = latest.merge(recent, on="sku_id")
    merged["days_of_supply"] = (
        (merged["inventory"] / merged["avg_daily_demand"].replace(0, np.nan))
        .fillna(0)
        .round(1)
    )

    def risk_color(dos, lt):
        if dos < lt:
            return "CRITICAL"
        if dos < 2 * lt:
            return "WARNING"
        return "OK"

    merged["risk"] = merged.apply(
        lambda r: risk_color(r["days_of_supply"], r["lead_time_days"]), axis=1
    )

    if category_filter != "All":
        merged = merged[merged["category"] == category_filter]

    color_map = {"CRITICAL": "#EA4335", "WARNING": "#FBBC05", "OK": "#34A853"}
    chart_font = dict(color="#111111", family="Segoe UI, Arial, sans-serif", size=13)

    # Chart 1: Days of Supply bar — sorted, color-coded by risk
    merged_sorted = merged.sort_values("days_of_supply")
    fig1 = go.Figure()
    for risk_val, color in color_map.items():
        sub = merged_sorted[merged_sorted["risk"] == risk_val]
        if sub.empty:
            continue
        fig1.add_trace(
            go.Bar(
                x=sub["sku_id"],
                y=sub["days_of_supply"],
                name=risk_val,
                marker_color=color,
                marker_line_color="#ffffff",
                marker_line_width=1,
                text=sub["days_of_supply"].round(1),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>Days of Supply: <b>%{y:.1f}</b><br><extra></extra>"
                ),
            )
        )
    fig1.update_layout(
        title=dict(
            text=f"Days of Supply by SKU — {category_filter}",
            font=dict(size=16, color="#4285F4"),
        ),
        xaxis_title="SKU ID",
        yaxis_title="Days of Supply",
        yaxis=dict(
            range=[
                0,
                merged_sorted["days_of_supply"].max() * 1.22,
            ],  # 22% headroom for labels
        ),
        barmode="group",
        height=440,
        legend=dict(title="Risk Level", orientation="h", y=1.08, x=0),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF",
        font=chart_font,
        xaxis=dict(tickangle=45, tickfont=dict(size=11)),
        margin=dict(t=80, b=80),
    )

    # Chart 2: Inventory vs Lead-Time Demand — grouped bar
    # Compare inventory against the units needed to bridge the reorder window,
    # NOT against 30-day demand (which makes short-lt SKUs look falsely critical).
    merged_sorted2 = (
        merged.sort_values("avg_daily_demand", ascending=False).head(20).copy()
    )
    merged_sorted2["lt_demand"] = (
        merged_sorted2["avg_daily_demand"] * merged_sorted2["lead_time_days"]
    ).round(0)

    color_map2 = {"CRITICAL": "#EA4335", "WARNING": "#F9AB00", "OK": "#34A853"}
    fig2 = go.Figure()
    for risk_val, color in color_map2.items():
        sub2 = merged_sorted2[merged_sorted2["risk"] == risk_val]
        if sub2.empty:
            continue
        fig2.add_trace(
            go.Bar(
                name=f"Inventory ({risk_val})",
                x=sub2["sku_id"],
                y=sub2["inventory"],
                marker_color=color,
                marker_line_color="#ffffff",
                marker_line_width=1,
                customdata=sub2[
                    ["days_of_supply", "lead_time_days", "avg_daily_demand"]
                ].values,
                hovertemplate=(
                    "<b>%{x}</b> [" + risk_val + "]<br>"
                    "Inventory: <b>%{y:,} units</b><br>"
                    "Days of Supply: <b>%{customdata[0]:.1f}d</b> vs Lead Time: %{customdata[1]:.0f}d<br>"
                    "Avg daily demand: %{customdata[2]:.1f}/day"
                    "<extra></extra>"
                ),
            )
        )
    fig2.add_trace(
        go.Bar(
            name="Lead-Time Demand (reorder buffer)",
            x=merged_sorted2["sku_id"],
            y=merged_sorted2["lt_demand"],
            marker_color="#94A3B8",
            marker_line_color="#ffffff",
            marker_line_width=1,
            customdata=merged_sorted2[["lead_time_days", "avg_daily_demand"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>Lead-Time Demand: <b>%{y:,} units</b><br>"
                "(%{customdata[1]:.1f}/day × %{customdata[0]:.0f}d lead time)<br>"
                "Inventory must stay above this line to avoid stockout"
                "<extra></extra>"
            ),
        )
    )
    n2_crit = int((merged_sorted2["risk"] == "CRITICAL").sum())
    n2_warn = int((merged_sorted2["risk"] == "WARNING").sum())
    fig2.update_layout(
        title=dict(
            text=(
                f"Inventory vs Lead-Time Demand — {category_filter}<br>"
                f"<sup style='color:#666'>{n2_crit} Critical &nbsp;|&nbsp; {n2_warn} Warning &nbsp;|&nbsp; "
                f"{len(merged_sorted2) - n2_crit - n2_warn} OK &nbsp;·&nbsp; "
                f"Grey = units needed to cover each SKU's reorder window</sup>"
            ),
            font=dict(size=15, color="#4285F4"),
        ),
        xaxis_title="SKU ID",
        yaxis_title="Units",
        barmode="group",
        height=460,
        legend=dict(orientation="h", y=1.10, x=0, font=dict(size=10)),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF",
        font=chart_font,
        xaxis=dict(tickangle=45, tickfont=dict(size=11)),
        margin=dict(t=90, b=80),
    )

    # KPI numbers
    n_critical = int((merged["risk"] == "CRITICAL").sum())
    n_warning = int((merged["risk"] == "WARNING").sum())
    n_ok = int((merged["risk"] == "OK").sum())
    avg_dos = float(merged["days_of_supply"].mean())

    # BUG-022 fix: guard supplier column so headers always align with data
    if "supplier" not in merged.columns:
        merged["supplier"] = "N/A"

    # At-risk table (WARNING + CRITICAL only)
    at_risk = (
        merged[merged["risk"] != "OK"]
        .sort_values("days_of_supply")[
            [
                "sku_id",
                "name",
                "category",
                "supplier",
                "inventory",
                "avg_daily_demand",
                "days_of_supply",
                "lead_time_days",
                "risk",
            ]
        ]
        .round({"avg_daily_demand": 1, "days_of_supply": 1})
    )

    # Full inventory table — all SKUs
    # BUG-046 fix: always include price_inr so column count matches the 10-column
    # headers declaration; fill with 0 when the column doesn't exist.
    if "price_inr" not in merged.columns:
        merged["price_inr"] = 0
    full_inv = merged.sort_values(["risk", "days_of_supply"])[
        [
            "sku_id",
            "name",
            "category",
            "supplier",
            "inventory",
            "avg_daily_demand",
            "days_of_supply",
            "lead_time_days",
            "price_inr",
            "risk",
        ]
    ].round({"avg_daily_demand": 1, "days_of_supply": 1})

    return (
        fig1,
        fig2,
        f"{n_critical}",
        f"{n_warning}",
        f"{n_ok}",
        f"{avg_dos:.1f}",
        at_risk,
        full_inv,
    )


def _get_store_choices() -> list[str]:
    """Build store dropdown choices from store_daily_inventory.csv."""
    try:
        sdi_path = BASE_DIR / "data" / "store_daily_inventory.csv"
        if sdi_path.exists():
            stores = pd.read_csv(
                sdi_path, usecols=["store_id", "city", "region", "store_type"]
            )
            stores = stores.drop_duplicates("store_id").sort_values("store_id")
            return ["All Stores"] + [
                f"{r['store_id']} — {r['city']} ({r['region']} · {r['store_type']})"
                for _, r in stores.iterrows()
            ]
    except Exception:
        pass
    return ["All Stores"]


def _parse_store_id(store_val: str) -> str | None:
    """
    Extract store_id from a dropdown value like 'ST001 — Mumbai (West · Flagship)'.
    Returns None when 'All Stores' is selected.
    Module-level so it can be used in build_inventory_tab, build_analytics_tab,
    and build_forecast_tab without NameError.
    """
    if not store_val or store_val == "All Stores":
        return None
    return store_val.split(" —")[0].strip()


def build_inventory_tab():
    with gr.TabItem("Inventory Dashboard"):
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#34A853,#0F9D58);
                    border-radius:12px; padding:18px 24px; margin-bottom:18px;">
            <div style="color:#fff; font-size:1.4rem; font-weight:800; margin-bottom:4px;">
                Real-Time Inventory Risk Dashboard
            </div>
            <div style="color:rgba(255,255,255,0.9); font-size:0.96rem;">
                Red = critical stock (below lead time). Yellow = warning. Green = healthy.
                Select a specific store to see per-store inventory.
            </div>
        </div>
        """)

        with gr.Row():
            view_dd = gr.Dropdown(
                choices=[
                    "Inventory Health Heatmap",
                    "Days of Supply",
                    "Inventory vs Demand",
                    "Stockout Risk Timeline",
                    "Dead Stock Analysis",
                ],
                value="Days of Supply",
                label="View",
                scale=2,
            )
            category_dd = gr.Dropdown(
                choices=[
                    "All",
                    "Dog Food",
                    "Dog Treats",
                    "Cat Food",
                    "Cat Treats",
                    "Cat Supplies",
                    "Health",
                    "Accessories",
                    "Toys",
                ],
                value="All",
                label="Filter by Category",
                scale=2,
            )
            store_dd = gr.Dropdown(
                choices=_get_store_choices(),
                value="All Stores",
                label="Filter by Store",
                scale=2,
            )
            refresh_btn = gr.Button("Refresh Dashboard", variant="primary", scale=1)

        # KPI cards as colored HTML boxes
        kpi_html = gr.HTML("""
        <div style="display:flex; gap:16px; margin:14px 0; flex-wrap:wrap;">
            <div style="flex:1; min-width:140px; background:#FCE8E6; border:2px solid #EA4335;
                        border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#EA4335;">—</div>
                <div style="font-size:0.9rem; font-weight:700; color:#C5221F;">CRITICAL SKUs</div>
            </div>
            <div style="flex:1; min-width:140px; background:#FEF9E0; border:2px solid #FBBC05;
                        border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#B06000;">—</div>
                <div style="font-size:0.9rem; font-weight:700; color:#B06000;">WARNING SKUs</div>
            </div>
            <div style="flex:1; min-width:140px; background:#E6F4EA; border:2px solid #34A853;
                        border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#137333;">—</div>
                <div style="font-size:0.9rem; font-weight:700; color:#137333;">OK SKUs</div>
            </div>
            <div style="flex:1; min-width:140px; background:#E8F0FE; border:2px solid #4285F4;
                        border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:#1967D2;">—</div>
                <div style="font-size:0.9rem; font-weight:700; color:#1967D2;">Avg Days of Supply</div>
            </div>
        </div>
        """)

        chart_main = gr.Plot(label="Inventory View")
        inv_interp = gr.Markdown("")

        _INV_INTERP = {
            "Inventory Health Heatmap": (
                "**What this shows:** Every SKU ranked by days of supply (shortest at top). "
                "Bar length = days of stock remaining.  \n"
                "**Colour = risk vs each SKU's OWN lead time:** "
                "Red = stock out before reorder arrives (DoS < lead time). "
                "Amber = running low (DoS < 2× lead time). Green = healthy.  \n"
                "**◆ red diamond** on each bar = that SKU's critical threshold (its own lead time). "
                "**▲ orange triangle** = warning threshold (2× lead time).  \n"
                "**Key insight:** A short green bar is safe if its red ◆ marker is to its left. "
                "SKUs with short lead times can safely hold less stock.  \n"
                "**Action:** Reorder all red bars immediately."
            ),
            "Days of Supply": (
                "**What this shows:** Each bar = how many days the current stock will last at the current rate of sales.  \n"
                "**Colour = risk relative to each SKU's OWN lead time:** "
                "Red = stock out before the next order arrives (DoS < lead time). "
                "Amber = running low (DoS < 2× lead time). Green = healthy.  \n"
                "**◆ red diamond** = that SKU's critical threshold (its own lead time). "
                "**▲ orange triangle** = that SKU's warning threshold (2× lead time).  \n"
                "**Important:** A short green bar is NOT a problem if the diamond marker is below it — "
                "that SKU has a short lead time and is genuinely safe.  \n"
                "**Action:** Reorder any red bar immediately."
            ),
            "Inventory vs Demand": (
                "**What this shows:** For each SKU, the coloured bar = current inventory (colour = risk level); "
                "the grey bar = lead-time demand (units needed to bridge the reorder window = avg daily demand × lead time).  \n"
                "**Why lead-time demand, not 30-day demand?** Risk is about whether you can survive until the next delivery arrives — "
                "not whether you have 30 days of stock. A SKU with a 2-day lead time only needs 2 days of stock before a reorder arrives. "
                "Comparing to 30-day demand would make all short-lead-time SKUs look falsely critical.  \n"
                "**How to read it:** When the coloured bar is shorter than the grey bar, "
                "inventory is below the minimum safe level for that SKU's lead time → the bar will be red (CRITICAL).  \n"
                "**Why is a short green bar still OK?** Because its lead time is short — "
                "a reorder can arrive quickly. The grey bar (reorder buffer) will be short too. Hover over any bar to see the exact numbers."
            ),
            "Stockout Risk Timeline": (
                "**What this shows:** A timeline of when each at-risk SKU is predicted to hit zero stock, "
                "based on current inventory and recent daily sales rate.  \n"
                "**How to read it:** Bars further left = stock out sooner. "
                "Red = will stock out before a new delivery can arrive (CRITICAL).  \n"
                "**Action:** Order the leftmost SKUs immediately."
            ),
            "Dead Stock Analysis": (
                "**What this shows:** For each SKU within each category, inventory value split into: "
                "Dead (no sales in 60+ days) shown in red, Slow-moving in amber, and Active stock in green. "
                "Values are in ₹INR (cost price).  \n"
                "**Action:** Plan clearance discounts for red SKUs to free up working capital."
            ),
        }

        def _fmt_inv_interp(view: str) -> str:
            return _INV_INTERP.get(view, "")

        gr.HTML(
            '<div style="margin-top:12px; font-size:1.1rem; font-weight:700; color:#EA4335;">At-Risk SKUs (Critical + Warning)</div>'
        )
        at_risk_table = gr.Dataframe(
            label="",
            headers=[
                "SKU",
                "Name",
                "Category",
                "Supplier",
                "Inventory",
                "Avg Demand/Day",
                "Days of Supply",
                "Lead Time",
                "Risk",
            ],
            interactive=False,
            wrap=True,
        )

        full_inv_header = gr.HTML(
            '<div style="margin-top:18px; font-size:1.1rem; font-weight:700; '
            'color:#4285F4;">Inventory Snapshot — All 65 SKUs</div>'
        )
        full_inv_table = gr.Dataframe(
            label="",
            headers=[
                "SKU",
                "Name",
                "Category",
                "Supplier",
                "Inventory",
                "Avg Demand/Day",
                "Days of Supply",
                "Lead Time",
                "Price",
                "Risk",
            ],
            interactive=False,
            wrap=True,
        )

        # _parse_store_id is defined at module level — available here and in other tabs

        def _get_store_df(store_id: str | None) -> pd.DataFrame | None:
            """Load store_daily_inventory.csv filtered to one store, or None for all-store view."""
            if store_id is None:
                return None
            try:
                sdi_path = BASE_DIR / "data" / "store_daily_inventory.csv"
                if not sdi_path.exists():
                    return None
                sdi = pd.read_csv(sdi_path, parse_dates=["date"])
                sdi = sdi[sdi["store_id"] == store_id]
                if sdi.empty:
                    return None
                latest = sdi["date"].max()
                sdi = sdi[sdi["date"] == latest].copy()
                # Rename to match the demand CSV schema expected by build_inventory_fig
                sdi = sdi.rename(columns={"demand": "demand", "date": "date"})
                return sdi
            except Exception:
                return None

        # ── Canonical risk computation used by BOTH KPIs and charts ────────
        # Single source of truth: all KPI boxes are derived from the same
        # merged DataFrame that the chart visualises, using the same risk
        # thresholds (CRITICAL: dos < lead_time; WARNING: dos < 2×lead_time).

        def _compute_inv_merged(cat: str, store_id: str | None) -> pd.DataFrame:
            """
            Return the canonical merged DataFrame for the given category and
            optional store filter.  KPIs and charts BOTH read from this.
            """
            if store_id:
                sdi_path = BASE_DIR / "data" / "store_daily_inventory.csv"
                if sdi_path.exists():
                    sdi = pd.read_csv(sdi_path, parse_dates=["date"])
                    sdi = sdi[sdi["store_id"] == store_id]
                    if not sdi.empty:
                        latest = sdi["date"].max()
                        sdi = sdi[sdi["date"] == latest].copy()
                        sdi["days_of_supply"] = sdi["days_of_supply"].fillna(0)

                        # Re-compute risk from the same formula used everywhere
                        # (pre-stored risk_status may use different thresholds)
                        def _risk(dos, lt):
                            if dos < lt:
                                return "CRITICAL"
                            if dos < 2 * lt:
                                return "WARNING"
                            return "OK"

                        sdi["risk"] = sdi.apply(
                            lambda r: _risk(r["days_of_supply"], r["lead_time_days"]),
                            axis=1,
                        )
                        if cat != "All":
                            sdi = sdi[sdi["category"] == cat]
                        # Normalise column names for downstream chart builders
                        if "name" not in sdi.columns and "sku_name" in sdi.columns:
                            sdi = sdi.rename(columns={"sku_name": "name"})
                        if (
                            "avg_daily_demand" not in sdi.columns
                            and "demand" in sdi.columns
                        ):
                            sdi["avg_daily_demand"] = sdi["demand"]
                        if "supplier" not in sdi.columns:
                            sdi["supplier"] = "N/A"
                        if "price_inr" not in sdi.columns:
                            sdi["price_inr"] = 0.0
                        return sdi
            # All-stores: use aggregated demand CSV (same as build_inventory_fig)
            df = get_df()
            latest_date = df["date"].max()
            latest = df[df["date"] == latest_date].copy()
            cutoff = latest_date - pd.Timedelta(days=30)
            recent = (
                df[df["date"] >= cutoff]
                .groupby("sku_id")["demand"]
                .mean()
                .reset_index()
                .rename(columns={"demand": "avg_daily_demand"})
            )
            merged = latest.merge(recent, on="sku_id")
            merged["days_of_supply"] = (
                (merged["inventory"] / merged["avg_daily_demand"].replace(0, np.nan))
                .fillna(0)
                .round(1)
            )

            def _risk(dos, lt):
                if dos < lt:
                    return "CRITICAL"
                if dos < 2 * lt:
                    return "WARNING"
                return "OK"

            merged["risk"] = merged.apply(
                lambda r: _risk(r["days_of_supply"], r["lead_time_days"]), axis=1
            )
            if cat != "All":
                merged = merged[merged["category"] == cat]
            if "supplier" not in merged.columns:
                merged["supplier"] = "N/A"
            if "price_inr" not in merged.columns:
                merged["price_inr"] = 0.0
            return merged

        def _pick_inv_chart(
            view: str, cat: str, store_id: str | None, merged: pd.DataFrame
        ) -> go.Figure:
            """
            Build the correct chart for `view`, using the canonical merged
            DataFrame so the chart and KPI boxes are always in sync.
            The merged DataFrame is passed in — no second data load.
            """
            if view == "Inventory Health Heatmap":
                # Horizontal bar: one row per SKU, length = days_of_supply
                return _build_heatmap_from_df(merged)
            elif view == "Stockout Risk Timeline":
                # Show only SKUs stocking out within 30 days
                return _build_stockout_from_df(merged, days_ahead=30)
            elif view == "Dead Stock Analysis":
                # Dead stock uses separate dead-stock logic; pass category filter
                return build_dead_stock_bar(cat)
            elif view == "Inventory vs Demand":
                # Grouped bar: current inventory vs 30-day demand
                return _build_inv_vs_demand_from_df(merged)
            else:  # "Days of Supply" (default)
                return _build_dos_from_df(merged)

        def _build_dos_from_df(merged: pd.DataFrame) -> go.Figure:
            """
            Days of Supply bar chart from the canonical merged df.

            Key design decisions:
            - Bar COLOUR encodes risk relative to each SKU's OWN lead time
              (CRITICAL: dos < lt, WARNING: dos < 2×lt, OK otherwise).
            - Per-SKU lead time is shown as a red diamond marker on each bar
              so users can see exactly which threshold applies to that SKU.
            - The old avg_lt reference lines are removed — they created false
              alarms (a short bar below the average line but above its own lt
              looked critical but was actually OK).
            """
            color_map = {"CRITICAL": "#EA4335", "WARNING": "#F9AB00", "OK": "#34A853"}
            merged_s = merged.sort_values("days_of_supply").reset_index(drop=True)
            cat_label = (
                merged["category"].iloc[0]
                if len(merged["category"].unique()) == 1
                else "All"
            )
            fig = go.Figure()
            # ── Coloured bars ─────────────────────────────────────────────────
            for risk_val, color in color_map.items():
                sub = merged_s[merged_s["risk"] == risk_val]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=sub["sku_id"],
                        y=sub["days_of_supply"],
                        name=risk_val,
                        marker_color=color,
                        marker_line_color="#ffffff",
                        marker_line_width=1,
                        text=sub["days_of_supply"].round(1),
                        textposition="outside",
                        customdata=np.column_stack(
                            [
                                sub[
                                    ["lead_time_days", "inventory", "avg_daily_demand"]
                                ].values,
                                (
                                    sub["lead_time_days"] * 2
                                ).values,  # [3] = 2×lt for hover
                            ]
                        ),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Days of Supply: <b>%{y:.1f}d</b><br>"
                            "Lead Time: <b>%{customdata[0]:.0f}d</b><br>"
                            "Critical if DoS &lt; <b>%{customdata[0]:.0f}d</b> &nbsp;|&nbsp; "
                            "Warning if DoS &lt; <b>%{customdata[3]:.0f}d</b> (2×LT)<br>"
                            "Inventory: %{customdata[1]:,} units &nbsp;|&nbsp; "
                            "Avg demand: %{customdata[2]:.1f}/day"
                            "<extra></extra>"
                        ),
                    )
                )

            # ── Per-SKU lead-time markers (red ◆ = critical threshold for that SKU) ──
            # This replaces the single avg_lt line that was misleading.
            fig.add_trace(
                go.Scatter(
                    x=merged_s["sku_id"],
                    y=merged_s["lead_time_days"],
                    mode="markers",
                    name="Lead Time (critical threshold)",
                    marker=dict(
                        symbol="diamond",
                        size=9,
                        color="#C5221F",
                        line=dict(color="#fff", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Lead Time: <b>%{y}d</b><br>"
                        "CRITICAL if DoS drops below this<extra></extra>"
                    ),
                )
            )

            # ── Per-SKU warning markers (orange ▲ = warning threshold = 2×lt) ─
            fig.add_trace(
                go.Scatter(
                    x=merged_s["sku_id"],
                    y=merged_s["lead_time_days"] * 2,
                    mode="markers",
                    name="2×Lead Time (warning threshold)",
                    marker=dict(
                        symbol="triangle-up",
                        size=8,
                        color="#F9AB00",
                        line=dict(color="#fff", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Warning threshold: <b>%{y}d</b> (2 × lead time)<br>"
                        "WARNING if DoS drops below this<extra></extra>"
                    ),
                )
            )

            n_crit = int((merged_s["risk"] == "CRITICAL").sum())
            n_warn = int((merged_s["risk"] == "WARNING").sum())
            fig.update_layout(
                **_CHART_LAYOUT,
                title=dict(
                    text=(
                        f"Days of Supply by SKU — {cat_label}<br>"
                        f"<sup style='color:#666'>"
                        f"{n_crit} Critical &nbsp;|&nbsp; {n_warn} Warning &nbsp;|&nbsp; "
                        f"{len(merged_s) - n_crit - n_warn} OK (of {len(merged_s)} SKUs) &nbsp;·&nbsp; "
                        f"◆ = each SKU's own lead-time threshold</sup>"
                    ),
                    font=dict(size=15, color="#4285F4"),
                ),
                xaxis_title="SKU ID",
                yaxis_title="Days of Supply",
                yaxis=dict(
                    range=[
                        0,
                        max(
                            merged_s["days_of_supply"].max(),
                            (merged_s["lead_time_days"] * 2).max(),
                        )
                        * 1.22,
                    ]
                ),
                barmode="group",
                height=440,
                legend=dict(
                    title="",
                    orientation="h",
                    y=1.10,
                    x=0,
                    font=dict(size=10),
                ),
                xaxis=dict(tickangle=45, tickfont=dict(size=11)),
                margin=dict(t=90, b=80),
            )
            return fig

        def _build_heatmap_from_df(merged: pd.DataFrame) -> go.Figure:
            """Inventory Health heatmap (horizontal bars) from the canonical df."""
            color_map = {"CRITICAL": "#EA4335", "WARNING": "#F9AB00", "OK": "#34A853"}
            risk_order = ["CRITICAL", "WARNING", "OK"]
            merged_s = merged.sort_values("days_of_supply", ascending=True).reset_index(
                drop=True
            )
            name_col = "name" if "name" in merged_s.columns else "sku_id"
            merged_s["label"] = merged_s["sku_id"] + " — " + merged_s[name_col].str[:20]
            # BUG-020: removed dead variable avg_lt (was computed but never used after
            # replacing single vline with per-SKU markers in a prior fix)
            fig = go.Figure()
            for risk in risk_order:
                sub = merged_s[merged_s["risk"] == risk]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        name=risk,
                        x=sub["days_of_supply"],
                        y=sub["label"],
                        orientation="h",
                        marker_color=color_map[risk],
                        marker_opacity=0.88,
                        text=[f"{d:.0f}d" for d in sub["days_of_supply"]],
                        textposition="outside",
                        textfont=dict(size=9),
                        customdata=sub[
                            ["lead_time_days", "inventory", "avg_daily_demand"]
                        ].values,
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Days of Supply: <b>%{x:.1f}d</b><br>"
                            "Lead Time: <b>%{customdata[0]:.0f}d</b> "
                            "(critical if < this, warning if < 2×this)<br>"
                            "Inventory: %{customdata[1]:,} units &nbsp;|&nbsp; "
                            "Avg demand: %{customdata[2]:.1f}/day"
                            "<extra></extra>"
                        ),
                    )
                )

            # Per-SKU lead-time markers — red ◆ on the bar at each SKU's own threshold
            fig.add_trace(
                go.Scatter(
                    x=merged_s["lead_time_days"],
                    y=merged_s["label"],
                    mode="markers",
                    name="Lead Time (critical threshold)",
                    marker=dict(
                        symbol="diamond",
                        size=9,
                        color="#C5221F",
                        line=dict(color="#fff", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>Lead Time: <b>%{x}d</b> (critical threshold)<extra></extra>"
                    ),
                )
            )
            # 2×lt warning markers — orange ▲
            fig.add_trace(
                go.Scatter(
                    x=merged_s["lead_time_days"] * 2,
                    y=merged_s["label"],
                    mode="markers",
                    name="2×Lead Time (warning threshold)",
                    marker=dict(
                        symbol="triangle-right",
                        size=8,
                        color="#F9AB00",
                        line=dict(color="#fff", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>Warning threshold: <b>%{x}d</b> (2 × lead time)<extra></extra>"
                    ),
                )
            )

            n_crit = int((merged_s["risk"] == "CRITICAL").sum())
            n_warn = int((merged_s["risk"] == "WARNING").sum())
            cat_label = (
                merged["category"].iloc[0]
                if len(merged["category"].unique()) == 1
                else "All"
            )
            fig.update_layout(
                **_CHART_LAYOUT,
                title=dict(
                    text=(
                        f"Inventory Health — {cat_label}<br>"
                        f"<sup style='color:#666'>{n_crit} Critical &nbsp;|&nbsp; "
                        f"{n_warn} Warning &nbsp;|&nbsp; "
                        f"{len(merged_s) - n_crit - n_warn} OK (of {len(merged_s)} SKUs) "
                        f"&nbsp;·&nbsp; ◆ = each SKU's own lead-time threshold</sup>"
                    ),
                    font=dict(size=14, color="#4285F4"),
                ),
                xaxis_title="Days of Supply remaining",
                yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
                height=max(440, len(merged_s) * 22 + 160),
                legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
                margin=dict(t=90, b=60, l=190, r=100),
                barmode="overlay",
            )
            return fig

        def _build_inv_vs_demand_from_df(merged: pd.DataFrame) -> go.Figure:
            """
            Inventory vs 30-Day Demand grouped bar from the canonical df.
            Inventory bars are colour-coded by risk (same as other views).
            A 'Reorder Buffer' marker shows the minimum safe inventory
            (avg_daily_demand × lead_time_days) so users can see how much
            headroom remains before reaching the critical threshold.
            """
            # Sort by demand desc, top 20 to keep chart readable
            top20 = (
                merged.sort_values("avg_daily_demand", ascending=False).head(20).copy()
            )
            top20["reorder_buffer"] = (
                top20["avg_daily_demand"] * top20["lead_time_days"]
            ).round(0)
            cat_label = (
                merged["category"].iloc[0]
                if len(merged["category"].unique()) == 1
                else "All"
            )
            color_map = {"CRITICAL": "#EA4335", "WARNING": "#F9AB00", "OK": "#34A853"}

            fig = go.Figure()
            # Inventory bars — coloured by risk
            for risk_val, color in color_map.items():
                sub = top20[top20["risk"] == risk_val]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        name=f"Inventory ({risk_val})",
                        x=sub["sku_id"],
                        y=sub["inventory"],
                        marker_color=color,
                        marker_opacity=0.85,
                        customdata=sub[
                            ["days_of_supply", "lead_time_days", "avg_daily_demand"]
                        ].values,
                        hovertemplate=(
                            "<b>%{x}</b> [" + risk_val + "]<br>"
                            "Inventory: <b>%{y:,} units</b><br>"
                            "Days of Supply: <b>%{customdata[0]:.1f}d</b><br>"
                            "Lead Time: %{customdata[1]:.0f}d<br>"
                            "Avg Daily Demand: %{customdata[2]:.1f}/day"
                            "<extra></extra>"
                        ),
                    )
                )

            # Lead-time demand bar: units needed to cover the reorder window
            # This is the CORRECT comparison — not 30-day demand.
            # A SKU with lt=2d only needs 2 days of stock before a reorder arrives.
            # Comparing to 30-day demand would make every short-lt SKU look critical.
            fig.add_trace(
                go.Bar(
                    name="Lead-Time Demand (units needed to bridge reorder)",
                    x=top20["sku_id"],
                    y=top20["reorder_buffer"],
                    marker_color="#94A3B8",
                    marker_opacity=0.6,
                    customdata=top20[["lead_time_days", "avg_daily_demand"]].values,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Lead-Time Demand: <b>%{y:,} units</b><br>"
                        "(avg %{customdata[1]:.1f}/day × %{customdata[0]:.0f}d lead time)<br>"
                        "Inventory must stay above this to avoid stockout"
                        "<extra></extra>"
                    ),
                )
            )

            # Use full merged dataset for counts so subtitle matches KPI boxes
            n_crit_all = int((merged["risk"] == "CRITICAL").sum())
            n_warn_all = int((merged["risk"] == "WARNING").sum())
            n_ok_all = int((merged["risk"] == "OK").sum())
            fig.update_layout(
                **_CHART_LAYOUT,
                title=dict(
                    text=(
                        f"Inventory vs Lead-Time Demand — {cat_label} (Top 20 by demand)<br>"
                        f"<sup style='color:#666'>"
                        f"{n_crit_all} Critical &nbsp;|&nbsp; "
                        f"{n_warn_all} Warning &nbsp;|&nbsp; "
                        f"{n_ok_all} OK across all {len(merged)} SKUs &nbsp;·&nbsp; "
                        f"Grey bar = units needed to bridge each SKU's own reorder window</sup>"
                    ),
                    font=dict(size=15, color="#4285F4"),
                ),
                xaxis_title="SKU ID",
                yaxis_title="Units",
                barmode="group",
                height=460,
                legend=dict(orientation="h", y=1.10, x=0, font=dict(size=10)),
                xaxis=dict(tickangle=45, tickfont=dict(size=11)),
                margin=dict(t=90, b=80),
            )
            return fig

        def _build_stockout_from_df(
            merged: pd.DataFrame, days_ahead: int = 30
        ) -> go.Figure:
            """Stockout Risk Timeline from the canonical df."""
            # days_of_supply IS days_until_stockout for this chart
            at_risk = (
                merged[merged["days_of_supply"] <= days_ahead]
                .sort_values("days_of_supply")
                .reset_index(drop=True)
            )
            if at_risk.empty:
                return _empty_fig(
                    f"No SKUs projected to stock out within {days_ahead} days — all well stocked!"
                )
            name_col = "name" if "name" in at_risk.columns else "sku_id"
            at_risk["label"] = at_risk["sku_id"] + " — " + at_risk[name_col].str[:22]

            def _color(d, lt):
                if d < lt:
                    return "#EA4335"
                if d <= 14:
                    return "#F9AB00"
                return "#4285F4"

            colors = [
                _color(d, lt)
                for d, lt in zip(at_risk["days_of_supply"], at_risk["lead_time_days"])
            ]

            fig = go.Figure()
            urgency_groups = [
                ("#EA4335", "CRITICAL — stocks out before reorder arrives"),
                ("#F9AB00", "WARNING — reorder urgent"),
                ("#4285F4", "MONITOR — some buffer remains"),
            ]
            for col, label in urgency_groups:
                idxs = [i for i, c in enumerate(colors) if c == col]
                if not idxs:
                    continue
                sub = at_risk.iloc[idxs]
                fig.add_trace(
                    go.Bar(
                        name=label,
                        x=sub["days_of_supply"],
                        y=sub["label"],
                        orientation="h",
                        marker_color=col,
                        marker_opacity=0.88,
                        text=[f"{d:.0f}d" for d in sub["days_of_supply"]],
                        textposition="outside",
                        textfont=dict(size=9),
                        hovertemplate=(
                            "<b>%{y}</b><br>Days until stockout: <b>%{x:.0f}d</b><extra></extra>"
                        ),
                    )
                )
            crit = sum(1 for c in colors if c == "#EA4335")
            fig.update_layout(
                **_CHART_LAYOUT,
                title=dict(
                    text=(
                        f"Stockout Risk — {len(at_risk)} SKUs running out within {days_ahead} days<br>"
                        f"<sup style='color:#EA4335'>{crit} CRITICAL (stock out before that SKU's own reorder can arrive) "
                        f"&nbsp;·&nbsp; Colour based on each SKU's individual lead time</sup>"
                    ),
                    font=dict(size=14, color="#EA4335"),
                ),
                xaxis_title="Days Until Stockout",
                yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
                height=max(420, len(at_risk) * 28 + 160),
                legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
                margin=dict(t=90, b=60, l=200, r=100),
                barmode="overlay",
            )
            return fig

        def update(view, cat, store_val):
            store_id = _parse_store_id(store_val)

            # ── Single canonical DataFrame for BOTH KPIs and chart ───────────
            # This guarantees KPI boxes always match exactly what the chart shows.
            merged = _compute_inv_merged(cat, store_id)

            if merged is None or merged.empty:
                # BUG-005 fix: must return exactly 6 values matching outs list
                return (
                    _empty_fig("No data available for this selection"),
                    "",
                    '<div style="padding:10px;color:#9CA3AF;">No inventory data for this selection.</div>',
                    [],
                    '<div style="margin-top:18px; font-size:1.1rem; font-weight:700; color:#9CA3AF;">No data available</div>',
                    [],
                )

            # ── KPI counts — derived from same DataFrame as chart ────────────
            c = int((merged["risk"] == "CRITICAL").sum())
            w = int((merged["risk"] == "WARNING").sum())
            ok = int((merged["risk"] == "OK").sum())
            dos = f"{merged['days_of_supply'].mean():.1f}d"
            n = (
                len(merged["sku_id"].unique())
                if "sku_id" in merged.columns
                else len(merged)
            )

            # Scope label
            store_label = (
                (store_val.split(" —")[1].strip() if " —" in store_val else store_id)
                if store_id
                else None
            )
            if store_label and cat != "All":
                scope = f"{store_label} · {cat} — {n} SKUs"
            elif store_label:
                scope = f"{store_label} — {n} SKUs"
            elif cat != "All":
                scope = f"{cat} — {n} SKUs"
            else:
                scope = f"All Stores — {n} SKUs"

            # ── At-risk table — same DataFrame ───────────────────────────────
            sup_col = "supplier" if "supplier" in merged.columns else "sku_id"
            name_col = "name" if "name" in merged.columns else "sku_id"
            adr = merged[merged["risk"] != "OK"].sort_values("days_of_supply")
            tbl_risk = (
                adr[
                    [
                        "sku_id",
                        name_col,
                        "category",
                        sup_col,
                        "inventory",
                        "avg_daily_demand",
                        "days_of_supply",
                        "lead_time_days",
                        "risk",
                    ]
                ]
                .rename(columns={name_col: "name", sup_col: "supplier"})
                .round({"avg_daily_demand": 1, "days_of_supply": 1})
            )

            # ── Full table — same DataFrame ───────────────────────────────────
            full = merged.sort_values(["risk", "days_of_supply"])
            price_col = "price_inr" if "price_inr" in full.columns else "inventory"
            tbl_full = (
                full[
                    [
                        "sku_id",
                        name_col,
                        "category",
                        sup_col,
                        "inventory",
                        "avg_daily_demand",
                        "days_of_supply",
                        "lead_time_days",
                        price_col,
                        "risk",
                    ]
                ]
                .rename(
                    columns={
                        name_col: "name",
                        sup_col: "supplier",
                        price_col: "price_inr",
                    }
                )
                .round({"avg_daily_demand": 1, "days_of_supply": 1})
            )

            # ── Chart — same DataFrame ────────────────────────────────────────
            fig = _pick_inv_chart(view, cat, store_id, merged)

            inv_hdr = (
                f'<div style="margin-top:18px; font-size:1.1rem; font-weight:700; '
                f'color:#4285F4;">Inventory Snapshot — {scope}</div>'
            )
            kpi = f"""
            <div style="display:flex; gap:16px; margin:14px 0; flex-wrap:wrap;">
                <div style="flex:1; min-width:140px; background:#FCE8E6; border:2px solid #EA4335;
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:2rem; font-weight:800; color:#EA4335;">{c}</div>
                    <div style="font-size:0.9rem; font-weight:700; color:#C5221F;">CRITICAL SKUs</div>
                </div>
                <div style="flex:1; min-width:140px; background:#FEF9E0; border:2px solid #FBBC05;
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:2rem; font-weight:800; color:#B06000;">{w}</div>
                    <div style="font-size:0.9rem; font-weight:700; color:#B06000;">WARNING SKUs</div>
                </div>
                <div style="flex:1; min-width:140px; background:#E6F4EA; border:2px solid #34A853;
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:2rem; font-weight:800; color:#137333;">{ok}</div>
                    <div style="font-size:0.9rem; font-weight:700; color:#137333;">OK SKUs</div>
                </div>
                <div style="flex:1; min-width:140px; background:#E8F0FE; border:2px solid #4285F4;
                            border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:2rem; font-weight:800; color:#1967D2;">{dos}</div>
                    <div style="font-size:0.9rem; font-weight:700; color:#1967D2;">Avg Days of Supply</div>
                </div>
            </div>
            """
            return fig, _fmt_inv_interp(view), kpi, tbl_risk, inv_hdr, tbl_full

        outs = [
            chart_main,
            inv_interp,
            kpi_html,
            at_risk_table,
            full_inv_header,
            full_inv_table,
        ]

        view_dd.change(update, inputs=[view_dd, category_dd, store_dd], outputs=outs)
        category_dd.change(
            update, inputs=[view_dd, category_dd, store_dd], outputs=outs
        )
        store_dd.change(update, inputs=[view_dd, category_dd, store_dd], outputs=outs)
        refresh_btn.click(update, inputs=[view_dd, category_dd, store_dd], outputs=outs)

        _ret = (category_dd, outs, update)

    # Return AFTER the TabItem context closes — avoids any context-manager edge-cases
    return _ret


def build_analytics_tab():
    """Analytics Dashboard tab with marketing, operational, and management charts."""
    with gr.TabItem("Analytics Dashboard"):
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#7C3AED 0%,#4F46E5 60%,#0891B2 100%);
                    border-radius:12px; padding:18px 24px; margin-bottom:18px;
                    box-shadow:0 4px 16px rgba(124,58,237,0.25);">
            <div style="color:#fff; font-size:1.4rem; font-weight:800; margin-bottom:4px;">
                Analytics Dashboard
            </div>
            <div style="color:rgba(255,255,255,0.9); font-size:0.96rem;">
                Marketing, operational, and management analytics for Pet Store supply chain.
            </div>
        </div>
        """)

        # ── Section 1: Marketing Analytics ──────────────────────────────────
        gr.HTML("""<div style="font-weight:800; font-size:18px; color:#4F46E5;
                               background:#EEF2FF; border-left:4px solid #4F46E5;
                               border-radius:6px; padding:10px 16px; margin:8px 0 12px 0;">
            Marketing Analytics</div>""")

        with gr.Row():
            mkt_chart_dd = gr.Dropdown(
                choices=[
                    "Sales by Channel",
                    "Brand Performance",
                    "Category Revenue Heatmap",
                    "Promotion Impact",
                    "Top SKUs by Revenue",
                    "Customer Segments",
                ],
                value="Sales by Channel",
                label="Chart Type",
                scale=2,
            )
            mkt_cat_dd = gr.Dropdown(
                choices=[
                    "All",
                    "Dog Food",
                    "Dog Treats",
                    "Cat Food",
                    "Cat Treats",
                    "Cat Supplies",
                    "Health",
                    "Accessories",
                    "Toys",
                ],
                value="All",
                label="Category Filter",
                scale=1,
            )
            mkt_channel_dd = gr.Dropdown(
                choices=["All", "Online", "Offline", "App"],
                value="All",
                label="Channel Filter",
                scale=1,
            )
            mkt_store_dd = gr.Dropdown(
                choices=_get_store_choices(),
                value="All Stores",
                label="Store Filter",
                scale=1,
            )
            mkt_period_sl = gr.Slider(
                minimum=30,
                maximum=365,
                step=30,
                value=90,
                label="Period (days)",
                scale=1,
            )
            mkt_run_btn = gr.Button("Update", variant="primary", scale=1)

        mkt_plot = gr.Plot(label="Marketing Chart", value=build_sales_by_channel(90))

        # Chart interpretation panel — gr.Markdown for correct left-alignment
        mkt_interp = gr.Markdown("")

        # Return plain Markdown strings — gr.Markdown renders them left-aligned natively
        def _interp_box(content: str) -> str:
            # Convert simple HTML formatting to Markdown
            return (
                content.replace("<b>", "**")
                .replace("</b>", "**")
                .replace("<br>", "  \n")
                .replace("<i>", "*")
                .replace("</i>", "*")
                .replace("&nbsp;", " ")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
            )

        _MKT_INTERP = {
            "Sales by Channel": _interp_box(
                "<b>How to read this chart:</b> A stacked area chart showing total revenue over time split by channel. "
                "Each coloured band = one sales channel (Online, Offline, App).<br>"
                "<b>What to look for:</b> A growing Online band means digital sales are increasing. "
                "Seasonal peaks = festival periods (Diwali, New Year). "
                "Use the **Period slider** to zoom into a specific time window."
            ),
            "Brand Performance": _interp_box(
                "<b>How to read this chart:</b> A horizontal bar chart ranking all brands by total revenue (bar length). "
                "The green diamond marker on the secondary axis shows gross margin %.<br>"
                "<b>Colours:</b> Blue = Pet Store Private Label | Red = Third Party brand.<br>"
                "<b>What to look for:</b> Brands with long bars but low margin diamonds = high revenue but low profit. "
                "Private Label brands (blue) at the top = healthy margin strategy. "
                "Use **Category Filter** to drill into a specific category."
            ),
            "Category Revenue Heatmap": _interp_box(
                "<b>How to read this chart:</b> A multi-line chart showing monthly revenue per category over time. "
                "Each coloured line = one product category. Yellow shaded bands = festival periods.<br>"
                "<b>What to look for:</b> Lines spiking upward during festivals = seasonal demand peaks — pre-stock those categories. "
                "Flat or declining lines = slow movers that may need promotions."
            ),
            "Promotion Impact": _interp_box(
                "<b>How to read this chart:</b> Each horizontal bar shows how much demand changed *during* a promotion vs the 7-day baseline before it.<br>"
                "<b>Bar colour:</b> Green = demand went UP | Red = demand went down or was flat.<br>"
                "<b>Dot:</b> Shows whether demand stayed elevated 14 days AFTER the promo ended.<br>"
                "<b>What to look for:</b> Long green bar + green dot = promotion created lasting habit (best outcome). "
                "Long green bar + red/yellow dot = demand pull-forward only (spike then drop — less valuable)."
            ),
            "Top SKUs by Revenue": _interp_box(
                "<b>How to read this chart:</b> Top 10 SKUs ranked by total revenue. Bar length = revenue. "
                "Colour = product category for quick grouping.<br>"
                "<b>What to look for:</b> SKUs with very high revenue should have healthy stock levels at all times. "
                "Use **Category Filter** to find top sellers per category. "
                "Use **Channel Filter** to see which products are online-only vs store bestsellers."
            ),
            "Customer Segments": _interp_box(
                "<b>How to read this chart:</b> Revenue contribution by customer segment (bars) and average order value (dotted red line on right axis).<br>"
                "<b>Segments:</b> Loyal Premium = high-LTV returners | New Pet Parent = recently acquired | "
                "Budget Conscious = price-sensitive | Multi-Pet Household = largest basket size.<br>"
                "<b>What to look for:</b> High revenue + low avg order value = frequent small purchases (loyalty-driven). "
                "High avg order value + lower revenue = infrequent but high-value buyers (upsell opportunity)."
            ),
        }

        def _update_mkt(chart_type, cat, channel, store_val, period):
            """Route to the correct chart builder passing ALL filters including store."""
            p = int(period)
            sid = _parse_store_id(store_val)  # module-level — handles None/All Stores
            if chart_type == "Sales by Channel":
                fig = build_sales_by_channel(period_days=p, store_id=sid)
            elif chart_type == "Brand Performance":
                fig = build_brand_performance_bubble(
                    category_filter=cat,
                    channel_filter=channel,
                    period_days=p,
                    store_id=sid,
                )
            elif chart_type == "Category Revenue Heatmap":
                fig = build_category_revenue_heatmap(
                    category_filter=cat,
                    channel_filter=channel,
                    period_days=p,
                    store_id=sid,
                )
            elif chart_type == "Promotion Impact":
                fig = build_promotion_impact(
                    category_filter=cat, channel_filter=channel, store_id=sid
                )
            elif chart_type == "Top SKUs by Revenue":
                fig = build_top_skus_bar(
                    category_filter=cat, channel_filter=channel, store_id=sid
                )
            elif chart_type == "Customer Segments":
                fig = build_customer_segment_donut(
                    category_filter=cat,
                    channel_filter=channel,
                    period_days=p,
                    store_id=sid,
                )
            else:
                fig = _empty_fig("Select a chart type")

            interp = _MKT_INTERP.get(chart_type, "")
            return fig, interp

        _mkt_inputs = [
            mkt_chart_dd,
            mkt_cat_dd,
            mkt_channel_dd,
            mkt_store_dd,
            mkt_period_sl,
        ]
        _mkt_outputs = [mkt_plot, mkt_interp]

        mkt_run_btn.click(_update_mkt, inputs=_mkt_inputs, outputs=_mkt_outputs)
        # Wire ALL filter changes so the chart updates immediately
        for _ctrl in [
            mkt_chart_dd,
            mkt_cat_dd,
            mkt_channel_dd,
            mkt_store_dd,
            mkt_period_sl,
        ]:
            _ctrl.change(_update_mkt, inputs=_mkt_inputs, outputs=_mkt_outputs)

        gr.HTML('<div style="border-top:1.5px solid #E2E8F0; margin:20px 0;"></div>')

        # ── Section 2: Operational Analytics ────────────────────────────────
        gr.HTML("""<div style="font-weight:800; font-size:18px; color:#0891B2;
                               background:#ECFEFF; border-left:4px solid #0891B2;
                               border-radius:6px; padding:10px 16px; margin:8px 0 12px 0;">
            Operational Analytics</div>""")

        with gr.Row():
            ops_chart_dd = gr.Dropdown(
                choices=[
                    "Lead Time Performance",
                    "Cold Chain Monitor",
                    "Seasonal Demand Index",
                    "Reorder Events Timeline",
                ],
                value="Lead Time Performance",
                label="Chart Type",
                scale=3,
            )
            ops_run_btn = gr.Button("Update", variant="primary", scale=1)

        ops_plot = gr.Plot(label="Operational Chart", value=build_lead_time_scatter())
        ops_interp = gr.Markdown("")

        _OPS_INTERP = {
            "Lead Time Performance": _interp_box(
                "<b>What this chart shows:</b> Each horizontal bar = one supplier's average on-time delivery %. "
                "Green = ≥95% OTD target. Amber = 88–95%. Red = below 88% (underperforming).<br>"
                "<b>Secondary axis:</b> Lead-time delta = actual − promised days. Positive (red) = delivering late. Negative (green) = delivering early.<br>"
                "<b>Target line:</b> Dashed green line at 95% OTD — the industry benchmark all suppliers should meet.<br>"
                "<b>What to act on:</b> Red OTD bars need supplier review. Positive lead-time delta = chronic lateness — raise in contract negotiation."
            ),
            "Cold Chain Monitor": _interp_box(
                "<b>Top panel — Weekly average temperature per SKU:</b> Each coloured line = one cold-chain SKU. "
                "Points are weekly averages (730 raw daily readings → ~104 weekly points) for a readable trend. "
                "The shaded band around each line shows the weekly min–max range. "
                "Dashed lines mark the safe zone boundaries: 0°C (min) and 6°C (max).<br>"
                "<b>Bottom panel — Monthly breach count:</b> Each bar = number of days that month where temperature was outside the 0°C–6°C safe zone. "
                "Tall bars = months with temperature control problems.<br>"
                "<b>What to act on:</b> Any SKU with increasing breach counts, or a weekly average trending toward 6°C, needs immediate cold storage review."
            ),
            "Seasonal Demand Index": _interp_box(
                "<b>What this chart shows:</b> A grouped bar chart showing how each of the top 6 categories' demand varies by month. "
                "Index 100 = that category's average month. Above 100 = above-average demand; below 100 = below-average demand.<br>"
                "<b>How to read it:</b> Each group of bars = one month. Each coloured bar = one product category. "
                "Tall bars in a month = peak season for that category.<br>"
                "<b>What to act on:</b> Any category with bars significantly above 100 in upcoming months = pre-stock now. "
                "Bars well below 100 = plan markdowns or reduce purchase orders for that period."
            ),
            "Reorder Events Timeline": _interp_box(
                "<b>What this chart shows:</b> How many purchase orders were placed each month, broken down by product category.<br>"
                "<b>How to read it:</b> Each bar = one month. Bar height = total reorder events. Different colours = different categories. "
                "Taller bars = more reactive buying (not planned ahead).<br>"
                "<b>What to act on:</b> Reorder spikes just before known festivals suggest reactive buying. "
                "Shifting to planned purchasing (2–3 months ahead) reduces cost and stockout risk."
            ),
        }

        def _update_ops(chart_type):
            if chart_type == "Lead Time Performance":
                fig = build_lead_time_scatter()
            elif chart_type == "Cold Chain Monitor":
                fig = build_cold_chain_trend()
            elif chart_type == "Seasonal Demand Index":
                fig = build_seasonal_demand_radar()
            elif chart_type == "Reorder Events Timeline":
                fig = build_reorder_events_timeline()
            else:
                fig = _empty_fig("Select a chart type")
            return fig, _OPS_INTERP.get(chart_type, "")

        ops_run_btn.click(
            _update_ops, inputs=[ops_chart_dd], outputs=[ops_plot, ops_interp]
        )
        ops_chart_dd.change(
            _update_ops, inputs=[ops_chart_dd], outputs=[ops_plot, ops_interp]
        )

        gr.HTML('<div style="border-top:1.5px solid #E2E8F0; margin:20px 0;"></div>')

        # ── Section 3: Management Dashboard ─────────────────────────────────
        gr.HTML("""<div style="font-weight:800; font-size:18px; color:#137333;
                               background:#E6F4EA; border-left:4px solid #137333;
                               border-radius:6px; padding:10px 16px; margin:8px 0 12px 0;">
            Management Dashboard</div>""")

        # Financial KPI cards — always visible, render on load.
        # Guard against missing CSV at first startup (H-06 fix).
        try:
            _fin_kpi_initial = build_financial_kpi_cards()
        except Exception:
            _fin_kpi_initial = (
                '<div style="padding:12px;color:#6B7280;font-size:0.9rem;">'
                "Financial KPIs loading... Run <code>python -m data.generate_data</code> "
                "to generate data, then refresh.</div>"
            )
        fin_kpi_html = gr.HTML(_fin_kpi_initial)

        with gr.Row():
            mgmt_chart_dd = gr.Dropdown(
                choices=[
                    "Private Label vs Third Party",
                    "Month-over-Month Growth",
                    "Store Inventory Comparison",
                ],
                value="Private Label vs Third Party",
                label="Chart Type",
                scale=2,
            )
            mgmt_store_dd = gr.Dropdown(
                choices=_get_store_choices(),
                value="All Stores",
                label="Store Filter",
                scale=2,
            )
            mgmt_run_btn = gr.Button("Update", variant="primary", scale=1)

        mgmt_plot = gr.Plot(
            label="Management Chart", value=build_private_label_vs_third_party()
        )
        mgmt_interp = gr.Markdown("")

        _MGMT_INTERP = {
            "Private Label vs Third Party": _interp_box(
                "<b>What this chart shows:</b> Revenue split between Pet Store's own Private Label products and Third Party brands, broken down by product category.<br>"
                "<b>How to read it:</b> Each horizontal bar = one category. Blue = Private Label revenue. Red = Third Party revenue. "
                "The % labels inside each segment show share within that category.<br>"
                "<b>What to look for:</b> Categories where Third Party dominates (long red bars) represent margin leakage — "
                "growing Private Label in those categories improves overall profitability."
            ),
            "Month-over-Month Growth": _interp_box(
                "<b>What this chart shows:</b> Revenue, units sold, and gross margin all indexed to 100 at the first month shown. "
                "Percentage labels = month-on-month revenue change.<br>"
                "<b>How to read it:</b> A line rising above 100 = growth vs baseline. A line falling below 100 = decline. "
                "Hover over any point to see the actual ₹ value for that month.<br>"
                "<b>What to look for:</b> Revenue growing faster than margin = pricing pressure or rising costs. "
                "Units growing faster than revenue = average selling price declining."
            ),
            "Store Inventory Comparison": _interp_box(
                "<b>What this chart shows:</b> Average days of supply per product category (aggregated across all SKUs in that category).<br>"
                "<b>How to read it:</b> Red = below that category's avg lead time (critical — can't restock before stockout). "
                "Amber = below 2× avg lead time (reorder soon). Green = well stocked. "
                "The red ◆ marker shows each category's own critical threshold (its avg lead time — not a fixed 7 days).<br>"
                "<b>What to act on:</b> Red categories need immediate reorder across all their SKUs."
            ),
        }

        def _update_mgmt(chart_type, store_val):
            sid = (
                _parse_store_id(store_val)
                if store_val and store_val != "All Stores"
                else None
            )
            if chart_type == "Private Label vs Third Party":
                fig = build_private_label_vs_third_party(store_id=sid)
            elif chart_type == "Month-over-Month Growth":
                fig = build_mom_growth_chart(store_id=sid)
            elif chart_type == "Store Inventory Comparison":
                fig = build_store_inventory_comparison(store_id=sid)
            else:
                fig = _empty_fig("Select a chart type")
            return fig, _MGMT_INTERP.get(chart_type, "")

        _mgmt_inputs = [mgmt_chart_dd, mgmt_store_dd]
        mgmt_run_btn.click(
            _update_mgmt, inputs=_mgmt_inputs, outputs=[mgmt_plot, mgmt_interp]
        )
        for _ctrl in [mgmt_chart_dd, mgmt_store_dd]:
            _ctrl.change(
                _update_mgmt, inputs=_mgmt_inputs, outputs=[mgmt_plot, mgmt_interp]
            )


# Tab 3: Demand Forecast


def build_forecast_fig(
    sku_id: str, horizon: int, compare_sku: str = "None", store_id: str | None = None
):
    # When a store is selected, use store-level demand from store_daily_inventory.csv
    # for more accurate per-store forecasting. Fall back to aggregated demand CSV.
    if store_id:
        try:
            sdi_path = BASE_DIR / "data" / "store_daily_inventory.csv"
            if sdi_path.exists():
                sdi = pd.read_csv(sdi_path, parse_dates=["date"])
                sku_id_clean = sku_id.split(" —")[0].strip()
                store_sku = sdi[
                    (sdi["store_id"] == store_id) & (sdi["sku_id"] == sku_id_clean)
                ]
                # Need at least 90 days of history for the forecast model encoder window.
                # store_daily_inventory.csv typically has only 1 snapshot per store per SKU.
                # Fall back to full demand CSV which has the full 730-day history.
                if not store_sku.empty and len(store_sku) >= 90:
                    df_src = store_sku.copy().sort_values("date")
                    logger.info(
                        f"[forecast] Using store {store_id} data ({len(store_sku)} rows)"
                    )
                else:
                    df_src = get_df()
                    # BUG-007: surface the fallback so user knows store data was insufficient
                    logger.warning(
                        f"[forecast] Store {store_id} has <90 rows for {sku_id}; using national demand data"
                    )
            else:
                df_src = get_df()
        except Exception:
            df_src = get_df()
    else:
        df_src = get_df()

    df = df_src
    sku_id = sku_id.split(" —")[0].strip()
    sku_df = df[df["sku_id"] == sku_id].sort_values("date")
    if sku_df.empty:
        return (
            go.Figure(),
            gr.update(visible=False),
            "",
            "",
            "SKU not found in dataset.",
        )

    name = sku_df["name"].iloc[-1]
    supplier = sku_df["supplier"].iloc[-1]
    category = sku_df["category"].iloc[-1]
    lead_time = int(sku_df["lead_time_days"].iloc[-1])
    # BUG-045 fix: use .iloc[-1] (most recent row) not .iloc[0] (oldest row) for price
    if "price_inr" in sku_df.columns:
        price = float(sku_df["price_inr"].iloc[-1])
    elif "price_usd" in sku_df.columns:
        price = float(sku_df["price_usd"].iloc[-1])
    else:
        price = 0.0
    inv = int(sku_df["inventory"].iloc[-1])

    # ── History: three separate figures so nothing is cramped ────────────
    # fig_hist  → Inventory area + risk zones + Reorder Point + Safety Stock
    # fig_demand → Daily demand bars + 14-day rolling average (full width below)
    # Both share the same date range; separate Y-axes avoid scale clash.
    history = sku_df.tail(90).copy()
    sigma_h = float(history["demand"].std())
    avg_d_h = float(history["demand"].mean())
    # ── Single consistent safety-stock and reorder-point computation ─────
    # Used for BOTH the chart annotation lines and the recommendation text.
    # Formula: ROP = avg_demand × lead_time + Z × σ × √(lead_time)  (Z=1.65 → 95%)
    ss = round(1.65 * sigma_h * float(np.sqrt(lead_time)))
    rop = round(avg_d_h * lead_time + ss)
    roll_avg = history["demand"].rolling(14, min_periods=1).mean()

    crit_level = avg_d_h * lead_time
    warn_level = avg_d_h * lead_time * 2
    inv_max = max(history["inventory"].max(), warn_level) * 1.12
    dates = history["date"].tolist()
    chart_font = dict(color="#202124", family="Segoe UI, Arial, sans-serif", size=12)

    # ── Figure 1: Inventory ───────────────────────────────────────────────
    fig_hist = go.Figure()

    # Risk zone bands
    fig_hist.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=[crit_level] * len(dates) + [0] * len(dates),
            fill="toself",
            fillcolor="rgba(234,67,53,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Critical Zone",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig_hist.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=[warn_level] * len(dates) + [crit_level] * len(dates),
            fill="toself",
            fillcolor="rgba(251,188,5,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Warning Zone",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig_hist.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=[inv_max] * len(dates) + [warn_level] * len(dates),
            fill="toself",
            fillcolor="rgba(52,168,83,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Safe Zone",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    # Inventory filled area
    fig_hist.add_trace(
        go.Scatter(
            x=dates,
            y=history["inventory"],
            mode="lines",
            name="Inventory",
            line=dict(color="#34A853", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(52,168,83,0.18)",
            hovertemplate="<b>%{x|%b %d}</b><br>Inventory: %{y:,} units<extra></extra>",
        )
    )
    # Threshold annotations
    fig_hist.add_hline(
        y=rop,
        line_dash="dash",
        line_color="#EA4335",
        line_width=1.8,
        annotation_text=f"Reorder Point ({rop:,})",
        annotation_position="right",
        annotation_font=dict(color="#EA4335", size=11),
    )
    fig_hist.add_hline(
        y=ss,
        line_dash="dot",
        line_color="#FBBC05",
        line_width=1.5,
        annotation_text=f"Safety Stock ({ss:,})",
        annotation_position="right",
        annotation_font=dict(color="#B06000", size=11),
    )
    fig_hist.update_layout(
        title=dict(
            text=f"Inventory Level — {sku_id} ({name})",
            font=dict(size=14, color="#34A853"),
        ),
        height=380,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF",
        font=chart_font,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        margin=dict(t=60, b=40, l=60, r=130),
        hovermode="x unified",
        yaxis=dict(title="Units in Stock", gridcolor="#E8EAED"),
        xaxis=dict(gridcolor="#E8EAED"),
    )

    # ── Figure 2: Daily Demand (full-width bar chart below both) ──────────
    fig_demand = go.Figure()
    fig_demand.add_trace(
        go.Bar(
            x=dates,
            y=history["demand"],
            name="Daily Demand",
            marker_color="#4285F4",
            opacity=0.65,
            hovertemplate="<b>%{x|%b %d}</b><br>Demand: %{y:.0f} units<extra></extra>",
        )
    )
    fig_demand.add_trace(
        go.Scatter(
            x=dates,
            y=roll_avg,
            mode="lines",
            name="14-day Avg",
            line=dict(color="#EA4335", width=2.2),
            hovertemplate="<b>%{x|%b %d}</b><br>14-day avg: %{y:.1f} units<extra></extra>",
        )
    )
    fig_demand.update_layout(
        title=dict(
            text="Daily Demand (last 90 days)",
            font=dict(size=14, color="#4285F4"),
        ),
        height=360,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF",
        font=chart_font,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)),
        margin=dict(t=50, b=60, l=60, r=30),
        hovermode="x unified",
        barmode="overlay",
        yaxis=dict(title="Units / Day", gridcolor="#E8EAED"),
        xaxis=dict(title="Date", gridcolor="#E8EAED"),
    )

    # ── Forecast engine: CatBoost first, statistical fallback ────────────
    last_date = sku_df["date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]
    method_label = "Statistical (mean ± 1.65σ)"

    try:
        from forecasting.ml_forecast import forecast as ml_forecast, is_trained

        if is_trained():
            preds = ml_forecast(sku_id, sku_df, horizon)
            p10 = preds["p10"]
            p50 = preds["p50"]
            p90 = preds["p90"]
            method_label = "CatBoost Quantile Regression"
        else:
            raise RuntimeError("not trained")
    except Exception:
        # BUG-036/038 fix: uncertainty bands widen with √t (proper forecast interval),
        # and trend is computed from actual recent data rather than fixed +5%.
        recent = sku_df["demand"].values[-60:]
        avg = float(np.mean(recent))
        std = float(np.std(recent)) if len(recent) > 1 else avg * 0.2
        # Compute actual linear trend from last 60 days
        if len(recent) >= 14:
            x = np.arange(len(recent), dtype=float)
            slope, intercept = np.polyfit(x, recent, 1)
            trend_per_day = slope / max(avg, 1e-6)  # fractional change per day
        else:
            trend_per_day = 0.0
        # Clip trend to ±0.5% per day to avoid extrapolation explosions
        trend_per_day = float(np.clip(trend_per_day, -0.005, 0.005))
        t = np.arange(1, horizon + 1, dtype=float)
        p50 = np.maximum(avg * (1.0 + trend_per_day * t), 0)
        # Uncertainty widens proportionally to √t (standard random-walk assumption)
        expanding_std = std * np.sqrt(t)
        p10 = np.maximum(p50 - 1.65 * expanding_std, 0)
        p90 = p50 + 1.65 * expanding_std

    fig_fore = go.Figure()
    fig_fore.add_trace(
        go.Scatter(
            x=future_dates,
            y=p90,
            mode="lines",
            name="P90 (Optimistic)",
            line=dict(color="#34A853", dash="dash", width=1.5),
            fill=None,
        )
    )
    fig_fore.add_trace(
        go.Scatter(
            x=future_dates,
            y=p50,
            mode="lines",
            name="P50 (Expected)",
            line=dict(color="#4285F4", width=2.5),
            fill="tonexty",
            fillcolor="rgba(66,133,244,0.10)",
        )
    )
    fig_fore.add_trace(
        go.Scatter(
            x=future_dates,
            y=p10,
            mode="lines",
            name="P10 (Pessimistic)",
            line=dict(color="#EA4335", dash="dash", width=1.5),
            fill="tonexty",
            fillcolor="rgba(234,67,53,0.08)",
        )
    )
    fig_fore.update_layout(
        title=f"{horizon}-Day Demand Forecast: {sku_id}  [{method_label}]",
        height=340,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#202124", family="Google Sans, Segoe UI, Arial, sans-serif"),
        title_font=dict(size=15, color="#4285F4"),
        xaxis_title="Date",
        yaxis_title="Daily Demand (units)",
    )

    # ── Reorder recommendation ────────────────────────────────────────────
    # Re-use the same ss / rop already computed for the chart annotations
    # so the chart lines and the text always show identical numbers.
    #
    # Order quantity: cover the forecast horizon demand, minus whatever
    # usable stock is already on hand (inventory above safety stock level).
    usable_inv = max(0, inv - ss)  # stock above safety buffer
    order_qty = max(0, int(p50.sum()) - usable_inv)  # only order what's missing

    if inv < rop and order_qty > 0:
        rec = (
            f"REORDER RECOMMENDED — current inventory ({inv:,} units) is below "
            f"the reorder point ({rop:,} units).\n"
            f"Suggested order: {order_qty:,} units  "
            f"(covers {horizon}-day P50 forecast of {int(p50.sum()):,} units "
            f"minus {usable_inv:,} usable units on hand).\n"
            f"Safety stock: {ss:,} units  |  Lead time: {lead_time} days."
        )
    elif inv < rop and order_qty == 0:
        rec = (
            f"Inventory ({inv:,} units) is below the reorder point ({rop:,} units), "
            f"but the {horizon}-day forecast demand is already covered by usable "
            f"stock on hand ({usable_inv:,} units above safety stock).\n"
            f"Monitor closely — consider ordering for the next cycle.\n"
            f"Safety stock: {ss:,} units  |  Lead time: {lead_time} days."
        )
    else:
        rec = (
            f"Stock adequate — inventory ({inv:,} units) is above the reorder "
            f"point ({rop:,} units).\n"
            f"Safety stock: {ss:,} units  |  Lead time: {lead_time} days."
        )

    p10_txt = f"{p10.sum():,.0f} units  ({p10.mean():.1f}/day)"
    p50_txt = f"{p50.sum():,.0f} units  ({p50.mean():.1f}/day)"
    p90_txt = f"{p90.sum():,.0f} units  ({p90.mean():.1f}/day)"

    meta = (
        f"**SKU:** {sku_id}  |  **Name:** {name}  |  **Category:** {category}\n"
        f"**Supplier:** {supplier}  |  **Lead Time:** {lead_time} days  |  **Price:** ₹{price:,.0f}  |  **Current Inv:** {inv:,}"
    )

    # Log prediction
    try:
        from mlops.monitor import log_prediction

        log_prediction(
            sku_id=sku_id,
            p10_total=float(p10.sum()),
            p50_total=float(p50.sum()),
            p90_total=float(p90.sum()),
            p50_daily=float(p50.mean()),
            horizon_days=horizon,
            forecast_source=method_label,
        )
    except Exception:
        pass

    # ── Unified chart: history + forecast continuation ────────────────────
    # BUG-025 fix: reuse future_dates already computed above (same values)
    chart_font = dict(color="#202124", family="Segoe UI, Arial, sans-serif", size=12)
    history = sku_df.tail(90).copy()
    hist_dates = history["date"].tolist()
    # future_dates is already defined above — no recomputation needed

    fig_unified = go.Figure()

    # Historical demand shaded area
    fig_unified.add_trace(
        go.Scatter(
            x=hist_dates,
            y=history["demand"],
            name="Historical Demand",
            mode="lines",
            line=dict(color="#4285F4", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(66,133,244,0.10)",
        )
    )

    # Inventory line on secondary y-axis
    fig_unified.add_trace(
        go.Scatter(
            x=hist_dates,
            y=history["inventory"],
            name="Inventory",
            mode="lines",
            line=dict(color="#9E9E9E", width=1, dash="dot"),
            yaxis="y2",
        )
    )

    # Vertical divider at last historical date — use add_shape to avoid
    # Plotly annotation bug with string x-values
    last_hist_str = str(sku_df["date"].iloc[-1])[:10]
    fig_unified.add_shape(
        type="line",
        x0=last_hist_str,
        x1=last_hist_str,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#BDBDBD", width=1.5, dash="dash"),
    )
    fig_unified.add_annotation(
        x=last_hist_str,
        y=1,
        yref="paper",
        text="▶ Forecast",
        showarrow=False,
        font=dict(size=10, color="#555"),
        xanchor="left",
        yanchor="top",
    )

    # P10–P90 confidence band
    fig_unified.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(p90) + list(p10[::-1]),
            fill="toself",
            fillcolor="rgba(234,67,53,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P10–P90 Band",
            hoverinfo="skip",
        )
    )

    # P50 forecast line
    fig_unified.add_trace(
        go.Scatter(
            x=future_dates,
            y=list(p50),
            name=f"P50 Forecast ({method_label})",
            mode="lines+markers",
            line=dict(color="#34A853", width=2.5),
            marker=dict(size=4),
        )
    )

    # P10 and P90 boundary lines
    fig_unified.add_trace(
        go.Scatter(
            x=future_dates,
            y=list(p10),
            name="P10 (Pessimistic)",
            mode="lines",
            line=dict(color="#4285F4", width=1.2, dash="dot"),
        )
    )
    fig_unified.add_trace(
        go.Scatter(
            x=future_dates,
            y=list(p90),
            name="P90 (Optimistic)",
            mode="lines",
            line=dict(color="#EA4335", width=1.2, dash="dot"),
        )
    )

    # Reorder point horizontal line
    ss_val = round(1.65 * float(history["demand"].std()) * float(np.sqrt(lead_time)))
    rop_val = round(float(history["demand"].mean()) * lead_time + ss_val)
    # BUG-6 fix: add_hline does not support yref="y2". Use add_shape which does.
    fig_unified.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=rop_val,
        y1=rop_val,
        yref="y2",
        line=dict(color="#FBBC05", width=1.5, dash="dash"),
    )
    fig_unified.add_annotation(
        x=0.01,
        xref="paper",
        y=rop_val,
        yref="y2",
        text=f"Reorder Point ({rop_val:,} units)",
        showarrow=False,
        font=dict(size=10, color="#92400E"),
        xanchor="left",
        bgcolor="rgba(255,255,255,0.7)",
    )

    # Promotion overlays in forecast window — use add_shape to avoid
    # Plotly annotation_position bug with datetime x-values
    try:
        promos = get_promotions_df()
        if not promos.empty:
            for _, promo in promos.iterrows():
                ps = pd.Timestamp(promo["start_date"])
                pe = pd.Timestamp(promo["end_date"])
                fd0 = pd.Timestamp(future_dates[0])
                fdn = pd.Timestamp(future_dates[-1])
                if ps <= fdn and pe >= fd0:
                    x0 = str(max(ps, fd0))[:10]
                    x1 = str(min(pe, fdn))[:10]
                    # Shaded band
                    fig_unified.add_shape(
                        type="rect",
                        x0=x0,
                        x1=x1,
                        y0=0,
                        y1=1,
                        yref="paper",
                        fillcolor="rgba(251,188,5,0.15)",
                        line_width=0,
                        layer="below",
                    )
                    # Label annotation
                    fig_unified.add_annotation(
                        x=x0,
                        y=0.97,
                        yref="paper",
                        text=str(promo["name"])[:16],
                        showarrow=False,
                        font=dict(size=9, color="#92400E"),
                        xanchor="left",
                        yanchor="top",
                        bgcolor="rgba(251,188,5,0.25)",
                    )
    except Exception:
        pass

    fig_unified.update_layout(
        **_CHART_LAYOUT,
        title=dict(
            text=f"{horizon}-Day Demand Forecast: {sku_id} — {name}",
            font=dict(size=16, color="#1e40af"),
        ),
        xaxis=dict(title="Date", tickangle=30),
        yaxis=dict(title="Demand (units)", side="left"),
        yaxis2=dict(
            title="Inventory (units)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
        height=520,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        margin=dict(t=80, b=100, l=60, r=60),
        hovermode="x unified",
    )

    # ── SKU accuracy indicator ────────────────────────────────────────────
    try:
        from mlops.monitor import get_sku_accuracy_chart

        acc_df = get_sku_accuracy_chart()
        sku_acc = (
            acc_df[acc_df["sku_id"] == sku_id] if not acc_df.empty else pd.DataFrame()
        )
        if not sku_acc.empty:
            err = float(sku_acc["abs_error_pct"].iloc[0])
            grade = sku_acc["grade"].iloc[0]
            grade_color = {
                "Excellent": "#137333",
                "Good": "#1967D2",
                "Fair": "#B45309",
                "Poor": "#C5221F",
            }.get(grade, "#374151")
            accuracy_html = f"""
            <div style="background:#F8F9FA;border-radius:8px;padding:10px 16px;
                        margin-bottom:10px;font-size:0.9rem;color:#111;">
                <b>Model Accuracy for {sku_id}:</b>
                Absolute Error = <b>{err:.1f}%</b> &nbsp;·&nbsp;
                Grade: <b style="color:{grade_color};">{grade}</b>
                &nbsp;·&nbsp; Based on last 30 days of actual vs predicted demand
            </div>"""
        else:
            accuracy_html = ""
    except Exception:
        accuracy_html = ""

    # ── Styled KPI cards ─────────────────────────────────────────────────
    # BUG-010 fix: use .iloc[-1] (most recent price), not .iloc[0] (oldest price)
    if "price_inr" in sku_df.columns:
        price_inr = float(sku_df["price_inr"].iloc[-1])
    elif "price_usd" in sku_df.columns:
        price_inr = float(sku_df["price_usd"].iloc[-1])
    else:
        price_inr = 0.0
    p50_total = float(p50.sum())
    p10_total = float(p10.sum())
    p90_total = float(p90.sum())
    p50_daily = float(p50.mean())
    conf_width = p90_total - p10_total
    p50_value_inr = p50_total * price_inr

    def _card(title, value, sub, color, bg):
        return f"""<div style="flex:1;min-width:130px;background:{bg};border-left:4px solid {color};
                    border-radius:0 8px 8px 0;padding:12px 14px;">
                <div style="font-weight:800;font-size:13px;color:{color};">{title}</div>
                <div style="font-size:20px;font-weight:800;color:#111;margin:4px 0;">{value}</div>
                <div style="font-size:11px;color:#555;">{sub}</div>
            </div>"""

    kpi_html = f"""<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0;">
        {_card("P50 Total", f"{int(p50_total):,} units", f"₹{_fmt_inr(p50_value_inr)} at ₹{price_inr:,.0f}/unit", "#137333", "#E6F4EA")}
        {_card("P50 Daily Avg", f"{p50_daily:.1f} units/day", f"Over {horizon}-day horizon", "#1967D2", "#EBF2FF")}
        {_card("P10 Pessimistic", f"{int(p10_total):,} units", "Lower bound (10th percentile)", "#4285F4", "#F0F4FF")}
        {_card("P90 Optimistic", f"{int(p90_total):,} units", "Upper bound (90th percentile)", "#C5221F", "#FCE8E6")}
        {_card("Confidence Width", f"{int(conf_width):,} units", f"P90−P10 = uncertainty range", "#92400E", "#FEF3C7")}
    </div>"""

    # ── Compare chart ─────────────────────────────────────────────────────
    show_compare = compare_sku and compare_sku != "None"
    if show_compare:
        try:
            sku2_id = compare_sku.split(" —")[0].strip()
            sku2_df = get_df()
            sku2_df = sku2_df[sku2_df["sku_id"] == sku2_id].sort_values("date")
            from forecasting.ml_forecast import (
                forecast as ml_forecast2,
                is_trained as _it2,
            )

            if _it2() and not sku2_df.empty:
                p2 = ml_forecast2(sku2_id, sku2_df, horizon)
                fig_compare = go.Figure()
                fig_compare.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=list(p50),
                        name=f"{sku_id} P50",
                        line=dict(color="#34A853", width=2.5),
                    )
                )
                fig_compare.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=list(p2["p50"]),
                        name=f"{sku2_id} P50",
                        line=dict(color="#4285F4", width=2.5),
                    )
                )
                # Uncertainty bands
                fig_compare.add_trace(
                    go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=list(p90) + list(p10[::-1]),
                        fill="toself",
                        fillcolor="rgba(52,168,83,0.10)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name=f"{sku_id} band",
                        hoverinfo="skip",
                    )
                )
                fig_compare.add_trace(
                    go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=list(p2["p90"]) + list(p2["p10"][::-1]),
                        fill="toself",
                        fillcolor="rgba(66,133,244,0.10)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name=f"{sku2_id} band",
                        hoverinfo="skip",
                    )
                )
                fig_compare.update_layout(
                    **_CHART_LAYOUT,
                    title=dict(
                        text=f"Comparison: {sku_id} vs {sku2_id} — P50 Forecast",
                        font=dict(size=15, color="#1e40af"),
                    ),
                    xaxis_title="Date",
                    yaxis_title="Demand (units)",
                    height=400,
                    hovermode="x unified",
                    legend=dict(orientation="h", y=-0.22, x=0),
                    margin=dict(t=70, b=90),
                )
                fig_compare_out = gr.update(value=fig_compare, visible=True)
            else:
                fig_compare_out = gr.update(visible=False)
        except Exception:
            fig_compare_out = gr.update(visible=False)
    else:
        fig_compare_out = gr.update(visible=False)

    return fig_unified, fig_compare_out, accuracy_html, kpi_html, rec


def build_forecast_tab():
    with gr.TabItem("Demand Forecast"):
        # ── Dynamic engine badge ──────────────────────────────────────────
        from forecasting.ml_forecast import (
            get_metrics as _fc_metrics,
            is_trained as _fc_ready,
        )

        _fc_meta = _fc_metrics()
        _fc_engine = _fc_meta.get("engine", "Not trained")
        _fc_mape = _fc_meta.get("mape", "—")
        _fc_badge = (
            "TFT"
            if "TFT" in _fc_engine
            else "CATBOOST"
            if "Cat" in _fc_engine
            else "PENDING"
        )
        gr.HTML(f"""
        <div style="background:linear-gradient(135deg,#FBBC05 0%,#F9AB00 100%);
                    border-radius:12px; padding:18px 24px; margin-bottom:16px;
                    box-shadow:0 4px 14px rgba(251,188,5,0.25);">
            <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                <div style="color:#fff; font-size:1.3rem; font-weight:800;
                            letter-spacing:-0.01em;">Per-SKU Demand Forecast</div>
                <span style="background:rgba(255,255,255,0.25); color:#fff;
                             font-size:0.78rem; font-weight:700; padding:3px 12px;
                             border-radius:20px; letter-spacing:0.05em;">{_fc_badge}</span>
            </div>
            <div style="color:rgba(255,255,255,0.92); font-size:0.9rem; margin-top:6px;">
                Engine: {_fc_engine} &nbsp;·&nbsp; P10 / P50 / P90 probabilistic forecast
                {"&nbsp;·&nbsp; Validation MAPE: " + str(_fc_mape) + "%" if _fc_mape != "—" else ""}
                &nbsp;·&nbsp; Promotion & festival features included
            </div>
        </div>
        """)

        # ── Build SKU lists ───────────────────────────────────────────────
        df = get_df()
        latest = (
            df[df["date"] == df["date"].max()][["sku_id", "name", "category"]]
            .drop_duplicates("sku_id")
            .sort_values("sku_id")
        )
        categories = ["All"] + sorted(latest["category"].dropna().unique().tolist())

        def _sku_choices(cat="All"):
            sub = latest if cat == "All" else latest[latest["category"] == cat]
            return [f"{r['sku_id']} — {r['name']}" for _, r in sub.iterrows()]

        all_choices = _sku_choices()

        # ── Controls row ──────────────────────────────────────────────────
        with gr.Row():
            cat_filter = gr.Dropdown(
                choices=categories,
                value="All",
                label="Category Filter",
                scale=1,
            )
            fcast_store_dd = gr.Dropdown(
                choices=_get_store_choices(),
                value="All Stores",
                label="Store Filter",
                scale=1,
            )
            sku_dd = gr.Dropdown(
                choices=all_choices,
                value=all_choices[0] if all_choices else None,
                label="Select SKU",
                scale=2,
            )
            sku_dd2 = gr.Dropdown(
                choices=["None"] + all_choices,
                value="None",
                label="Compare with SKU (optional)",
                scale=2,
            )

        with gr.Row():
            gr.HTML("""<div style="display:flex;gap:8px;align-items:center;padding:4px 0;">
                <span style="font-weight:600;font-size:0.9rem;color:#374151;">Horizon Presets:</span>
            </div>""")
            preset_7 = gr.Button("1 Week", size="sm", scale=1)
            preset_14 = gr.Button("2 Weeks", size="sm", scale=1)
            preset_30 = gr.Button("1 Month", size="sm", scale=1)
            preset_90 = gr.Button("3 Months", size="sm", scale=1)
            horizon_sl = gr.Slider(
                minimum=7,
                maximum=90,
                step=1,
                value=30,
                label="Custom Horizon (days)",
                scale=3,
            )
            run_btn = gr.Button("Run Forecast", variant="primary", scale=1)

        # ── Quantile legend cards ─────────────────────────────────────────
        gr.HTML("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 14px 0;">
            <div style="flex:1;min-width:140px;background:#EBF2FF;border-left:4px solid #4285F4;
                        border-radius:0 8px 8px 0;padding:10px 14px;">
                <div style="font-weight:800;font-size:13px;color:#1967D2;">P10 — Pessimistic</div>
                <div style="font-size:11px;color:#374151;margin-top:3px;">
                    10% chance demand falls below this. Use for conservative planning.</div>
            </div>
            <div style="flex:1;min-width:140px;background:#E6F4EA;border-left:4px solid #34A853;
                        border-radius:0 8px 8px 0;padding:10px 14px;">
                <div style="font-weight:800;font-size:13px;color:#137333;">P50 — Expected</div>
                <div style="font-size:11px;color:#374151;margin-top:3px;">
                    Median forecast — best number for daily demand planning.</div>
            </div>
            <div style="flex:1;min-width:140px;background:#FCE8E6;border-left:4px solid #EA4335;
                        border-radius:0 8px 8px 0;padding:10px 14px;">
                <div style="font-weight:800;font-size:13px;color:#C5221F;">P90 — Optimistic</div>
                <div style="font-size:11px;color:#374151;margin-top:3px;">
                    90% chance demand stays below this. Use for safety stock.</div>
            </div>
            <div style="flex:1;min-width:140px;background:#FEF3C7;border-left:4px solid #F59E0B;
                        border-radius:0 8px 8px 0;padding:10px 14px;">
                <div style="font-weight:800;font-size:13px;color:#92400E;">Promotions</div>
                <div style="font-size:11px;color:#374151;margin-top:3px;">
                    Orange shading = active promotion in the forecast window.</div>
            </div>
        </div>
        """)

        # ── Chart interpretation — gr.Markdown for correct left-alignment ──
        gr.Markdown(
            "**How to read the Demand Forecast chart:**  \n"
            "The chart is split into two sections by a vertical dashed line. "
            "**Left of the line** = the last 90 days of actual historical demand (blue line) "
            "and inventory (grey dotted). **Right of the line** = the forecast period.  \n"
            "The **green line** (P50) is the expected demand — your best planning number. "
            "The **shaded red band** between P10 and P90 shows the uncertainty range — "
            "demand is expected to stay within this band 80% of the time. "
            "**Orange shaded blocks** = active promotions during the forecast window "
            "(expect higher demand during those periods).  \n"
            "The **KPI cards below** show the total forecast in units and estimated revenue at the current price. "
            "The **yellow dashed horizontal line** on the right Y-axis shows the "
            "Reorder Point — if inventory drops below this, a purchase order is needed."
        )

        # ── SKU accuracy indicator ────────────────────────────────────────
        accuracy_html = gr.HTML("")

        # ── Main unified chart (history + forecast) ───────────────────────
        fig_unified = gr.Plot(label="Demand History + Forecast")

        # ── Comparison chart (shown only if compare SKU selected) ─────────
        fig_compare = gr.Plot(label="SKU Comparison", visible=False)

        # ── KPI cards ─────────────────────────────────────────────────────
        kpi_html = gr.HTML("")

        # ── Recommendation ────────────────────────────────────────────────
        rec_box = gr.Textbox(label="Reorder Recommendation", interactive=False, lines=3)

        # ── CSV export ────────────────────────────────────────────────────
        with gr.Row():
            export_btn = gr.Button("Export Forecast CSV", variant="secondary", scale=1)
            export_file = gr.File(label="Download", scale=2, visible=False)

        # ── Wire category + store filter → SKU list ──────────────────────
        def _update_sku_list(cat, store_val="All Stores"):
            sid = _parse_store_id(store_val)  # now module-level — always in scope
            if sid:
                # Filter SKUs available at this specific store
                try:
                    sdi_path = BASE_DIR / "data" / "store_daily_inventory.csv"
                    if sdi_path.exists():
                        sdi = pd.read_csv(
                            sdi_path,
                            usecols=["store_id", "sku_id", "category"],
                        )
                        store_skus = sdi[sdi["store_id"] == sid]["sku_id"].unique()
                        if len(store_skus) > 0:
                            sub = latest[latest["sku_id"].isin(store_skus)].copy()
                            if cat != "All" and not sub.empty:
                                cat_sub = sub[sub["category"] == cat]
                                # If category not stocked at this store, show all store SKUs
                                sub = cat_sub if not cat_sub.empty else sub
                            choices = [
                                f"{r['sku_id']} — {r['name']}"
                                for _, r in sub.iterrows()
                            ]
                            if choices:
                                return gr.update(choices=choices, value=choices[0])
                except Exception:
                    pass  # any failure → fall through to unfiltered list
            choices = _sku_choices(cat)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        cat_filter.change(
            _update_sku_list, inputs=[cat_filter, fcast_store_dd], outputs=[sku_dd]
        )
        fcast_store_dd.change(
            _update_sku_list, inputs=[cat_filter, fcast_store_dd], outputs=[sku_dd]
        )

        # ── Horizon preset buttons (BUG-021 fix: also re-run forecast) ─────
        _forecast_inputs = [sku_dd, sku_dd2, horizon_sl, fcast_store_dd]

        def _run_forecast(sku_raw, sku2_raw, horizon, store_val="All Stores"):
            sid = (
                _parse_store_id(store_val)
                if store_val and store_val != "All Stores"
                else None
            )
            return build_forecast_fig(
                sku_raw, int(horizon), compare_sku=sku2_raw, store_id=sid
            )

        _forecast_outputs = [fig_unified, fig_compare, accuracy_html, kpi_html, rec_box]

        def _set_horizon_and_run(h, sku_raw, sku2_raw):
            return (h,) + build_forecast_fig(sku_raw, int(h), compare_sku=sku2_raw)

        for _preset_btn, _h in [
            (preset_7, 7),
            (preset_14, 14),
            (preset_30, 30),
            (preset_90, 90),
        ]:
            _preset_btn.click(
                lambda sku_raw, sku2_raw, store_val, h=_h: (
                    (h,)
                    + build_forecast_fig(
                        sku_raw,
                        h,
                        compare_sku=sku2_raw,
                        store_id=_parse_store_id(store_val)
                        if store_val and store_val != "All Stores"
                        else None,
                    )
                ),
                inputs=[sku_dd, sku_dd2, fcast_store_dd],
                outputs=[horizon_sl] + _forecast_outputs,
            )

        run_btn.click(
            _run_forecast,
            inputs=_forecast_inputs,
            outputs=_forecast_outputs,
        )
        # BUG-020 fix: wire horizon slider change to re-run forecast
        horizon_sl.change(
            _run_forecast,
            inputs=_forecast_inputs,
            outputs=_forecast_outputs,
        )
        # Auto-run when SKU or store changes
        sku_dd.change(
            _run_forecast,
            inputs=_forecast_inputs,
            outputs=_forecast_outputs,
        )
        fcast_store_dd.change(
            _run_forecast,
            inputs=_forecast_inputs,
            outputs=_forecast_outputs,
        )

        # ── Export callback ───────────────────────────────────────────────
        def _export_forecast(sku_raw, horizon):
            import tempfile, csv as _csv

            sku_id = sku_raw.split(" —")[0].strip()
            df2 = get_df()
            sku_df = df2[df2["sku_id"] == sku_id].sort_values("date")
            if sku_df.empty:
                return gr.update(visible=False)
            try:
                from forecasting.ml_forecast import forecast as ml_forecast, is_trained

                if not is_trained():
                    return gr.update(visible=False)
                preds = ml_forecast(sku_id, sku_df, int(horizon))
            except Exception:
                return gr.update(visible=False)

            last_date = sku_df["date"].iloc[-1]
            rows = []
            for i in range(int(horizon)):
                fd = last_date + pd.Timedelta(days=i + 1)
                rows.append(
                    {
                        "date": fd.strftime("%Y-%m-%d"),
                        "sku_id": sku_id,
                        "p10": round(float(preds["p10"][i]), 1),
                        "p50": round(float(preds["p50"][i]), 1),
                        "p90": round(float(preds["p90"][i]), 1),
                    }
                )
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8"
            )
            writer = _csv.DictWriter(
                tmp, fieldnames=["date", "sku_id", "p10", "p50", "p90"]
            )
            writer.writeheader()
            writer.writerows(rows)
            tmp.close()
            return gr.update(value=tmp.name, visible=True)

        export_btn.click(
            _export_forecast,
            inputs=[sku_dd, horizon_sl],
            outputs=[export_file],
        )


# Tab 4: MLOps Monitor


def build_mlops_tab(
    mysql_host,
    mysql_port,
    mysql_user,
    mysql_password,
    mysql_db,
    pg_host,
    pg_port,
    pg_user,
    pg_password,
    pg_db,
):
    with gr.TabItem("MLOps Monitor"):
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#EA4335 0%,#C5221F 100%);
                    border-radius:12px; padding:20px 26px; margin-bottom:16px;
                    box-shadow:0 4px 14px rgba(234,67,53,0.25);">
            <div style="color:#fff; font-size:1.3rem; font-weight:800;
                        letter-spacing:-0.01em; margin-bottom:6px;">
                MLOps Monitor
            </div>
            <div style="color:rgba(255,255,255,0.92); font-size:0.9rem;">
                Retrain the TFT or CatBoost model, track forecasts, and detect drift.
            </div>
        </div>
        """)

        # ── Model Training Section ────────────────────────────────────────
        gr.HTML("""<div style="font-weight:800; font-size:19px; color:#1e40af;
                               background:#EFF6FF; border-left:4px solid #2563eb;
                               border-radius:6px; padding:10px 16px;
                               margin-bottom:12px;">Model Training — TFT Demand Forecasting</div>""")

        # Engine info banner — load cache first so status is accurate at build time
        from forecasting.ml_forecast import (
            load_models as _load_ml_models,
            get_metrics as _get_ml_metrics,
            is_trained as _is_ml_trained,
        )

        _load_ml_models()  # ensure cache is loaded before reading status

        _cur_metrics = _get_ml_metrics()
        _engine = _cur_metrics.get("engine", "not_trained")
        _mode_label = _cur_metrics.get("mode", "")

        if _is_ml_trained():
            _metric_parts = " &nbsp;·&nbsp; ".join(
                filter(
                    None,
                    [
                        f"<b>Engine:</b> <span style='color:#137333;font-weight:700'>{_engine}</span>",
                        f"<b>Mode:</b> {_mode_label}" if _mode_label else "",
                        f"<b>MAPE:</b> {_cur_metrics['mape']}%"
                        if _cur_metrics.get("mape") is not None
                        else "",
                        f"<b>MAE:</b> {_cur_metrics['mae']}"
                        if _cur_metrics.get("mae") is not None
                        else "",
                        f"<b>RMSE:</b> {_cur_metrics['rmse']}"
                        if _cur_metrics.get("rmse") is not None
                        else "",
                        f"<b>Calibration:</b> {_cur_metrics['calibration_pct']}%"
                        if _cur_metrics.get("calibration_pct") is not None
                        else "",
                        f"<b>Trained:</b> {_cur_metrics['trained_at']}"
                        if _cur_metrics.get("trained_at")
                        else "",
                    ],
                )
            )
            _engine_html = f"""
            <div style="background:#F0FDF4; border:1px solid #34A853; border-radius:8px;
                        padding:14px 20px; margin-bottom:12px; font-size:0.9rem;
                        color:#111111; line-height:1.8;">
                {_metric_parts}
            </div>"""
        else:
            _engine_html = """
            <div style="background:#FFF8E1; border:1px solid #F59E0B; border-radius:8px;
                        padding:14px 20px; margin-bottom:12px; font-size:0.9rem;
                        color:#5F3300; line-height:1.7;">
                <b>No model trained yet.</b> Click <b>Full TFT Retrain</b> to train on your GPU
                (~20–40 min), or <b>Fine-tune TFT</b> if a checkpoint exists (~3–8 min).
                <b>CatBoost Fallback</b> trains in ~2 min and is always available.
            </div>"""

        gr.HTML(_engine_html)

        # Training mode explanation — native Gradio Markdown (no custom HTML)
        gr.Markdown(
            "**Full TFT Retrain** — Trains the Temporal Fusion Transformer from scratch on "
            "all historical data. Uses GPU with BF16 mixed precision if available, CPU otherwise. "
            "Takes ~20–40 minutes on GPU, longer on CPU. Run this weekly or when new SKUs are added.  \n"
            "**Fine-tune TFT** — Loads the existing TFT checkpoint and continues training "
            "on the last 90 days of data. Takes ~3–8 minutes. Run this anytime new data arrives "
            "or after a major promotion to update the model without full retraining.  \n"
            "**CatBoost Fallback** — Fast tabular model. Trains in ~2 minutes. Used "
            "automatically when TFT is not yet trained. Includes promotion + festival features."
        )

        with gr.Row():
            train_source = gr.Radio(
                choices=["CSV (HUFT)", "MySQL", "PostgreSQL"],
                value="CSV (HUFT)",
                label="Training data source",
                info="CSV uses huft_daily_demand.csv with all promotion features.",
                scale=2,
            )

        with gr.Row():
            full_retrain_btn = gr.Button(
                "Full TFT Retrain (GPU, ~30 min)",
                variant="primary",
                scale=1,
            )
            finetune_btn = gr.Button(
                "Fine-tune TFT (~5 min)",
                variant="secondary",
                scale=1,
            )
            catboost_btn = gr.Button(
                "CatBoost Fallback (~2 min)",
                variant="secondary",
                scale=1,
            )

        train_status = gr.Textbox(
            label="Training status",
            interactive=False,
            lines=4,
            value=(
                f"Engine: {_engine} | "
                f"MAPE: {_cur_metrics.get('mape', '?')}% | "
                f"MAE: {_cur_metrics.get('mae', '?')} | "
                f"RMSE: {_cur_metrics.get('rmse', '?')} | "
                f"Calibration: {_cur_metrics.get('calibration_pct', '?')}% | "
                f"Trained: {_cur_metrics.get('trained_at', '—')} | "
                f"SKUs: {_cur_metrics.get('n_skus', '?')}"
            )
            if _is_ml_trained()
            else "No model trained. Click a training button above to start.",
        )

        def _run_training(
            source_label,
            fine_tune,
            use_catboost,
            mh,
            mp,
            mu,
            mpw,
            mdb,
            ph,
            pp,
            pu,
            ppw,
            pdb,
        ):
            src_map = {"CSV (HUFT)": "csv", "MySQL": "mysql", "PostgreSQL": "postgres"}
            src = src_map.get(source_label, "csv")
            mysql_c = {"host": mh, "port": mp, "user": mu, "password": mpw, "db": mdb}
            pg_c = {"host": ph, "port": pp, "user": pu, "password": ppw, "db": pdb}

            if use_catboost:
                result = _init_ml(
                    source=src,
                    mysql_creds=mysql_c if src == "mysql" else None,
                    pg_creds=pg_c if src == "postgres" else None,
                    fine_tune=False,
                    force_catboost=True,
                )
            else:
                result = _init_ml(
                    source=src,
                    mysql_creds=mysql_c if src == "mysql" else None,
                    pg_creds=pg_c if src == "postgres" else None,
                    fine_tune=fine_tune,
                    force_catboost=False,
                )

            if "error" in result:
                return f"Training failed: {result['error']}"

            engine = result.get("engine", "Unknown")
            mode = result.get("mode", "")
            return (
                f"Engine: {engine} ({mode}) | "
                f"MAPE: {result.get('mape', '?')}% | "
                f"MAE: {result.get('mae', '?')} | "
                f"RMSE: {result.get('rmse', '?')} | "
                f"Calibration: {result.get('calibration_pct', '?')}% | "
                f"Trained: {result.get('trained_at', '?')} | "
                f"Rows: {result.get('train_rows', 0):,} | "
                f"SKUs: {result.get('n_skus', '?')}"
            )

        _db_inputs = [
            mysql_host,
            mysql_port,
            mysql_user,
            mysql_password,
            mysql_db,
            pg_host,
            pg_port,
            pg_user,
            pg_password,
            pg_db,
        ]

        full_retrain_btn.click(
            lambda src, *db: _run_training(src, False, False, *db),
            inputs=[train_source] + _db_inputs,
            outputs=[train_status],
        )
        finetune_btn.click(
            lambda src, *db: _run_training(src, True, False, *db),
            inputs=[train_source] + _db_inputs,
            outputs=[train_status],
        )
        catboost_btn.click(
            lambda src, *db: _run_training(src, False, True, *db),
            inputs=[train_source] + _db_inputs,
            outputs=[train_status],
        )

        gr.HTML('<div style="border-top:1px solid #E2E8F0; margin:20px 0;"></div>')
        gr.HTML("""<div style="font-weight:800; font-size:19px; color:#1e40af;
                               background:#EFF6FF; border-left:4px solid #2563eb;
                               border-radius:6px; padding:10px 16px;
                               margin-bottom:12px;">Forecast Logs</div>""")

        with gr.Row():
            refresh_btn = gr.Button("Refresh Logs", variant="secondary")
            drift_btn = gr.Button("Run Drift Check", variant="primary")
            clear_log_btn = gr.Button("Clear Log", variant="secondary")

        # Summary stats
        with gr.Row():
            stat_total = gr.Textbox(label="Total Forecasts Logged", interactive=False)
            stat_skus = gr.Textbox(label="Unique SKUs Queried", interactive=False)
            stat_last = gr.Textbox(label="Last Forecast At", interactive=False)

        drift_output = gr.HTML(label="")

        chart_queried = gr.Plot(label="Forecast Accuracy by SKU")

        pred_log_table = gr.Dataframe(
            label="Latest Prediction per SKU (deduplicated — one row per SKU)",
            interactive=False,
        )

        query_log_table = gr.Dataframe(
            label="Query Log (last 50)",
            interactive=False,
        )

        def refresh_logs():
            from mlops.monitor import (
                get_prediction_log,
                get_query_log,
                get_forecast_summary,
                get_sku_accuracy_chart,
            )

            pred_df_all = get_prediction_log(10000)  # read all for summary stats
            query_df = get_query_log(50)
            summary = get_forecast_summary()
            acc_df = get_sku_accuracy_chart(top_n=20)

            # Deduplicate: keep only the most recent prediction per SKU.
            # pred_df_all is already sorted newest-first by get_prediction_log.
            if not pred_df_all.empty:
                pred_df = pred_df_all.drop_duplicates(
                    subset=["sku_id"], keep="first"
                ).reset_index(drop=True)
            else:
                pred_df = pred_df_all

            # ── SKU Forecast Accuracy chart ──────────────────────────────
            fig = go.Figure()
            if not acc_df.empty:
                # Colour bars by accuracy grade
                colour_map = {
                    "Excellent": "#34A853",
                    "Good": "#4285F4",
                    "Fair": "#FBBC05",
                    "Poor": "#EA4335",
                }
                colours = acc_df["grade"].map(colour_map).fillna("#9CA3AF").tolist()
                fig.add_trace(
                    go.Bar(
                        x=acc_df["sku_id"],
                        y=acc_df["abs_error_pct"],
                        marker_color=colours,
                        text=acc_df["grade"],
                        textposition="outside",
                        customdata=acc_df[["p50_daily", "actual_daily"]].values,
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Error: %{y:.1f}%<br>"
                            "Forecast P50: %{customdata[0]:.1f}/day<br>"
                            "Actual avg: %{customdata[1]:.1f}/day"
                            "<extra></extra>"
                        ),
                    )
                )
                # Reference lines for grade thresholds
                for level, colour, label in [
                    (10, "#34A853", "Excellent <10%"),
                    (20, "#4285F4", "Good <20%"),
                    (35, "#FBBC05", "Fair <35%"),
                ]:
                    fig.add_hline(
                        y=level,
                        line_dash="dot",
                        line_color=colour,
                        line_width=1,
                        annotation_text=label,
                        annotation_position="right",
                        annotation_font=dict(size=10, color=colour),
                    )
                fig.update_layout(
                    title=dict(
                        text="Forecast Accuracy per SKU  (lower = better)",
                        font=dict(size=14, color="#4285F4"),
                    ),
                    yaxis_title="Absolute Error %",
                    xaxis_title="SKU",
                    height=340,
                    plot_bgcolor="#F8F9FA",
                    paper_bgcolor="#FFFFFF",
                    font=dict(
                        color="#202124",
                        family="Segoe UI, Arial, sans-serif",
                        size=12,
                    ),
                    xaxis=dict(tickangle=45),
                    margin=dict(t=60, b=80, r=120),
                    yaxis=dict(
                        range=[0, max(acc_df["abs_error_pct"].max() * 1.3, 40)],
                    ),
                )
            else:
                fig.update_layout(
                    title="Run at least one forecast to see accuracy data",
                    height=340,
                    plot_bgcolor="#F8F9FA",
                    paper_bgcolor="#FFFFFF",
                )

            return (
                str(summary["total_forecasts"]),
                str(summary["unique_skus"]),
                str(summary.get("last_forecast_at", "—"))[:19],
                fig,
                pred_df,
                query_df,
            )

        def run_drift():
            from mlops.monitor import compute_drift_metrics

            r = compute_drift_metrics()

            # ── Render as HTML cards instead of raw JSON ──────────────────
            drift_ok = not r.get("drift_detected", False)
            status_bg = "#E6F4EA" if drift_ok else "#FCE8E6"
            status_bdr = "#34A853" if drift_ok else "#EA4335"
            status_txt = "#137333" if drift_ok else "#C5221F"
            status_lbl = "No Drift Detected" if drift_ok else "DRIFT DETECTED"

            worst_rows = "".join(
                f"<tr style='border-bottom:1px solid #E2E8F0;'>"
                f"<td style='padding:8px 12px;font-weight:600;'>{s.get('sku_id', '')}</td>"
                f"<td style='padding:8px 12px;'>{float(s.get('p50_daily', 0)):.2f}</td>"
                f"<td style='padding:8px 12px;'>{float(s.get('actual_daily_avg', 0)):.2f}</td>"
                f"<td style='padding:8px 12px;color:#EA4335;font-weight:700;'>"
                f"{float(s.get('abs_error', 0)):.2f}</td>"
                f"</tr>"
                for s in (r.get("worst_skus") or [])
            )
            worst_table = (
                f"""
            <table style='width:100%;border-collapse:collapse;margin-top:8px;'>
              <thead><tr style='background:#EBF2FF;'>
                <th style='padding:8px 12px;text-align:left;color:#1967D2;'>SKU</th>
                <th style='padding:8px 12px;text-align:left;color:#1967D2;'>P50 Forecast/day</th>
                <th style='padding:8px 12px;text-align:left;color:#1967D2;'>Actual/day</th>
                <th style='padding:8px 12px;text-align:left;color:#1967D2;'>Abs Error</th>
              </tr></thead>
              <tbody>{worst_rows}</tbody>
            </table>"""
                if worst_rows
                else "<p style='color:#9CA3AF;'>No SKU-level data available.</p>"
            )

            html = f"""
            <div style="border:2px solid {status_bdr};background:{status_bg};
                        border-radius:12px;padding:14px 20px;margin-bottom:16px;">
                <div style="font-weight:800;font-size:1.1rem;color:{status_txt};">
                    {status_lbl}
                </div>
            </div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
              <div style="flex:1;min-width:130px;background:#F8FAFC;border:1px solid #DDE3EA;
                          border-radius:10px;padding:12px 16px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:#4285F4;">
                    {r.get("forecast_mae", 0):.2f}</div>
                <div style="font-size:12px;color:#5F6368;margin-top:2px;">Forecast MAE</div>
              </div>
              <div style="flex:1;min-width:130px;background:#F8FAFC;border:1px solid #DDE3EA;
                          border-radius:10px;padding:12px 16px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:#9CA3AF;">
                    {r.get("baseline_mae", 0):.2f}</div>
                <div style="font-size:12px;color:#5F6368;margin-top:2px;">Baseline MAE</div>
              </div>
              <div style="flex:1;min-width:130px;background:#F8FAFC;border:1px solid #DDE3EA;
                          border-radius:10px;padding:12px 16px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:#34A853;">
                    {r.get("calibration_pct", 0):.0f}%</div>
                <div style="font-size:12px;color:#5F6368;margin-top:2px;">Calibration</div>
              </div>
              <div style="flex:1;min-width:130px;background:#F8FAFC;border:1px solid #DDE3EA;
                          border-radius:10px;padding:12px 16px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:#FBBC05;">
                    {r.get("n_evaluated", 0)}</div>
                <div style="font-size:12px;color:#5F6368;margin-top:2px;">Evaluated</div>
              </div>
            </div>
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;">
                Worst-performing SKUs</div>
            {worst_table}
            """
            return html

        _log_outputs = [
            stat_total,
            stat_skus,
            stat_last,
            chart_queried,
            pred_log_table,
            query_log_table,
        ]

        refresh_btn.click(refresh_logs, outputs=_log_outputs)
        drift_btn.click(run_drift, outputs=[drift_output])

        def clear_log():
            """Delete predictions.csv and return a refreshed (empty) view."""
            from mlops.monitor import PRED_LOG_PATH

            try:
                if PRED_LOG_PATH.exists():
                    PRED_LOG_PATH.unlink()
            except Exception:
                pass
            return refresh_logs()

        clear_log_btn.click(clear_log, outputs=_log_outputs)


# Main UI builder

# Minimal CSS — only targets our own elem_classes that the theme API cannot reach.
# Everything else (colours, borders, typography, buttons) is controlled via
# gr.themes.Default().set(...) below.
MINIMAL_CSS = """
/* ── Force light mode (override system dark preference) ─────────────── */
html, body { color-scheme: light !important; }

/* ── Page margin: 48px on both sides ────────────────────────────────── */
.gradio-container > .main > .wrap {
    padding-left: 48px !important;
    padding-right: 48px !important;
    box-sizing: border-box !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* ── Database panel: force same 48px inset so it never overflows ─────── */
#db-panel-col,
#db-panel-col > *,
#db-panel-col .gap,
#db-panel-col .form,
#db-panel-col .block {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}
/* The inner flex row (MySQL + PostgreSQL side by side) */
#db-panel-col .row,
#db-panel-col [class*="flex"] {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}

/* ── Tab navigation ─────────────────────────────────────────────────── */
button[role="tab"] {
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    letter-spacing: 0.01em !important;
}


/* ── Hide chatbot copy / action buttons ─────────────────────────────── */
.message-buttons, .copy-btn, .copy-button,
[aria-label="Copy"], [title="Copy"],
.full-width-chat button[aria-label="copy"],
.chatbot-bubble-buttons, .bot-actions { display: none !important; }

/* ── Chatbot: expand to full window width ───────────────────────────── */
.full-width-chat .bubble-wrap,
.full-width-chat .message-wrap { max-width: 100% !important; width: 100% !important; }
.full-width-chat .bubble-wrap .bubble,
.full-width-chat .message,
.full-width-chat .bot,
.full-width-chat .user { max-width: 96% !important; width: 96% !important; }

/* ── Chatbot message font ───────────────────────────────────────────── */
.full-width-chat .prose p, .full-width-chat .prose li,
.full-width-chat .prose { font-size: 16px !important; line-height: 1.75 !important; }

/* ── Dataframe table text ───────────────────────────────────────────── */
.table-wrap td, .table-wrap th,
.gradio-dataframe td, .gradio-dataframe th { font-size: 14px !important; }

/* ── Equal-height input row: all three boxes same width & height ────── */
.llm-config-row { align-items: stretch !important; }
.llm-config-row > * { flex: 1 1 0 !important; min-width: 0 !important; }

/* ── Database Settings row: pin button height to match heading ──────── */
.db-toggle-btn { height: 48px !important; min-height: 48px !important;
                 max-height: 48px !important; align-self: center !important; }

/* ── Database panel: constrain to page width, no overflow ───────────── */
.db-panel,
.db-panel > .block,
.db-panel > div,
.db-panel .gap,
.db-panel .form {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}
.db-panel .row,
.db-panel [class*="row"] {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    flex-wrap: wrap !important;
    box-sizing: border-box !important;
}
.db-panel input,
.db-panel textarea,
.db-panel select,
.db-panel .wrap,
.db-panel label {
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}
"""

# JS injected at page-load: force light mode + directly style elements that
# CSS selectors can't reliably reach in Gradio 6 (accordion button, tab labels).
_FORCE_LIGHT_JS = """
() => {
    const html = document.documentElement;
    html.classList.remove('dark');
    html.setAttribute('data-color-scheme', 'light');
    html.style.colorScheme = 'light';

    function fixOverflow() {
        const panel = document.getElementById('db-panel-col');
        if (!panel) return;

        /* Constrain the panel to page width with matching margin */
        panel.style.setProperty('width', '100%', 'important');
        panel.style.setProperty('max-width', '100%', 'important');
        panel.style.setProperty('box-sizing', 'border-box', 'important');
        panel.style.setProperty('overflow-x', 'hidden', 'important');

        /* Target the fields row specifically — remove gap and force it to fit */
        const row = document.getElementById('db-fields-row');
        if (row) {
            row.style.setProperty('width', '100%', 'important');
            row.style.setProperty('max-width', '100%', 'important');
            row.style.setProperty('box-sizing', 'border-box', 'important');
            row.style.setProperty('overflow-x', 'hidden', 'important');
            row.style.setProperty('gap', '8px', 'important');
            row.style.setProperty('flex-wrap', 'nowrap', 'important');
        }
        const mysqlCol = document.getElementById('db-mysql-col');
        if (mysqlCol) {
            mysqlCol.style.setProperty('min-width', '0', 'important');
            mysqlCol.style.setProperty('flex', '1 1 0', 'important');
            mysqlCol.style.setProperty('max-width', '50%', 'important');
        }
        /* All children: no element wider than 100% */
        panel.querySelectorAll('*').forEach(el => {
            el.style.setProperty('max-width', '100%', 'important');
            el.style.setProperty('min-width', '0', 'important');
            el.style.setProperty('box-sizing', 'border-box', 'important');
        });
    }
    fixOverflow();
    setTimeout(fixOverflow, 300);
    setTimeout(fixOverflow, 1000);
    /* Debounced observer — only watches the db-panel-col element, not the whole body.
       Fires at most once every 200ms to avoid thrashing during active agent sessions. */
    let _overflowTimer = null;
    const _debouncedFix = () => {
        if (_overflowTimer) return;
        _overflowTimer = setTimeout(() => { fixOverflow(); _overflowTimer = null; }, 200);
    };
    const panel = document.getElementById('db-panel-col');
    if (panel) {
        const observer = new MutationObserver(_debouncedFix);
        observer.observe(panel, { childList: true, subtree: false, attributes: false });
    }
}
"""


def _build_theme() -> gr.themes.Default:
    """
    Default theme with explicit light background and generous font sizes.
    Default theme renders labels as plain text (not Soft's coloured badges),
    so block_label_text_size is actually respected by the browser.
    """
    return gr.themes.Default(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        # Force light background regardless of system dark-mode
        body_background_fill="#F0F4F8",
        body_text_color="#0f172a",
        body_text_size="16px",
        # Blocks / cards
        block_background_fill="#ffffff",
        block_border_color="#cbd5e1",
        block_border_width="1.5px",
        block_radius="12px",
        block_shadow="0 2px 8px rgba(0,0,0,0.07)",
        # Labels — large, bold, dark
        block_label_background_fill="#EFF6FF",
        block_label_border_color="#BFDBFE",
        block_label_text_size="15px",
        block_label_text_weight="700",
        block_label_text_color="#1e40af",
        block_title_text_size="17px",
        block_title_text_weight="700",
        block_title_text_color="#0f172a",
        # Inputs
        input_background_fill="#ffffff",
        input_border_color="#cbd5e1",
        input_border_width="1.5px",
        input_radius="8px",
        input_placeholder_color="#94a3b8",
        input_border_color_focus="#2563eb",
        input_shadow_focus="0 0 0 3px rgba(37,99,235,0.15)",
        # Buttons
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#2563eb",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#EFF6FF",
        button_secondary_text_color="#2563eb",
        button_secondary_border_color="#2563eb",
        button_large_radius="8px",
        button_small_radius="6px",
        # Tables
        table_even_background_fill="#F8FAFF",
        table_odd_background_fill="#ffffff",
        table_border_color="#e2e8f0",
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pet Store Supply Chain Intelligence") as demo:
        # ── Main header ────────────────────────────────────────────────────
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#1e40af 0%,#2563eb 60%,#0891b2 100%);
                    border-radius:14px; padding:28px 36px; margin-bottom:20px;
                    box-shadow:0 4px 24px rgba(37,99,235,0.25);">
            <div style="margin-bottom:8px;">
                <h1 style="margin:0; font-size:1.7rem; font-weight:800; color:#ffffff;
                           letter-spacing:-0.02em; line-height:1.2;">
                    Pet Store Supply Chain Intelligence
                </h1>
            </div>
            <p style="margin:0; color:rgba(255,255,255,0.85); font-size:1rem; line-height:1.5;">
                AI-powered inventory management &nbsp;·&nbsp; Demand forecasting
                &nbsp;·&nbsp; ReAct agent with 50 MCP tools
            </p>
        </div>
        """)

        # ── LLM Configuration ──────────────────────────────────────────────
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#7c3aed 0%,#4f46e5 60%,#0ea5e9 100%);
                    border-radius:12px; padding:16px 24px; margin:8px 0 16px 0;
                    box-shadow:0 3px 16px rgba(124,58,237,0.22);">
            <div style="color:#fff; font-size:1.2rem; font-weight:800; margin-bottom:4px;">
                LLM Configuration
            </div>
            <div style="color:rgba(255,255,255,0.87); font-size:0.92rem; line-height:1.5;">
                Choose your AI provider, model, and enter your API key (or leave blank if set in .env).
            </div>
        </div>
        """)

        # Compute initial model list from agent.PROVIDERS — single source of truth
        from agent.agent import PROVIDERS as _PROVIDERS

        _default_prov = os.getenv("DEFAULT_PROVIDER", "anthropic").lower()
        _initial_model_choices = _PROVIDERS.get(_default_prov, _PROVIDERS["anthropic"])[
            "models"
        ]
        _default_model_val = os.getenv(
            "DEFAULT_MODEL",
            _PROVIDERS.get(_default_prov, _PROVIDERS["anthropic"])["default_model"],
        )
        if _default_model_val not in _initial_model_choices:
            _default_model_val = _initial_model_choices[0]

        with gr.Row(elem_classes=["llm-config-row"]):
            provider_dd = gr.Dropdown(
                choices=["Anthropic", "OpenAI", "Groq", "Gemini"],
                value=os.getenv("DEFAULT_PROVIDER", "anthropic").capitalize(),
                label="LLM Provider",
                scale=1,
            )
            model_dd = gr.Dropdown(
                choices=_initial_model_choices,
                value=_default_model_val,
                label="Model  (auto-updates with provider)",
                scale=1,
            )
            api_key_box = gr.Textbox(
                value="",
                label="API Key  (session only — never stored)",
                type="password",
                placeholder="Enter API Key, or leave blank to use .env",
                scale=1,
            )

        key_status_box = gr.Textbox(
            value=_key_status(os.getenv("DEFAULT_PROVIDER", "anthropic"), ""),
            label="Key Status",
            interactive=False,
            max_lines=1,
            elem_id="key-status-box",
        )

        def update_models_and_status(provider, api_key):
            from agent.agent import PROVIDERS

            prov_lower = provider.lower().strip()
            models = PROVIDERS.get(prov_lower, {}).get("models", [])
            default = PROVIDERS.get(prov_lower, {}).get(
                "default_model", models[0] if models else ""
            )
            return gr.update(choices=models, value=default), _key_status(
                provider, api_key
            )

        provider_dd.change(
            update_models_and_status,
            inputs=[provider_dd, api_key_box],
            outputs=[model_dd, key_status_box],
        )
        api_key_box.change(
            _key_status,
            inputs=[provider_dd, api_key_box],
            outputs=[key_status_box],
        )

        # Database Settings Panel — HTML header + toggled Column (no CSS needed)
        with gr.Row(equal_height=True):
            gr.HTML("""
            <div style="font-size:20px; font-weight:800; color:#ffffff;
                        background:linear-gradient(135deg,#1e40af 0%,#2563eb 100%);
                        border-radius:10px; padding:0 22px; flex:1;
                        display:flex; align-items:center; height:48px;
                        box-shadow:0 2px 10px rgba(37,99,235,0.18);">
                Database Settings &mdash; MySQL + PostgreSQL
            </div>""")
            _db_toggle_btn = gr.Button(
                "Show",
                variant="secondary",
                size="sm",
                scale=0,
                min_width=90,
                elem_classes=["db-toggle-btn"],
            )

        _db_open = gr.State(False)
        with gr.Column(
            visible=False, elem_classes=["db-panel"], elem_id="db-panel-col"
        ) as _db_panel:
            gr.HTML("""
            <style>
                /* Injected directly — overrides Gradio theme regardless of load order */
                .db-panel, .db-panel > *, .db-panel .gap, .db-panel .form,
                .db-panel .block, .db-panel .row, .db-panel [class*="row"],
                .db-panel > div, .db-panel > div > div {
                    max-width: 100% !important;
                    width: 100% !important;
                    min-width: 0 !important;
                    box-sizing: border-box !important;
                    overflow-x: hidden !important;
                }
            </style>
            <div style="background:linear-gradient(135deg,#FBBC05 0%,#F9AB00 100%);
                        border-radius:10px; padding:14px 20px; margin-bottom:16px;">
                <div style="color:#fff; font-size:1.05rem; font-weight:800; margin-bottom:2px;">
                    Database Connection
                </div>
                <div style="color:rgba(255,255,255,0.92); font-size:0.9rem;">
                    Connect to MySQL and PostgreSQL. Switch between Local and Cloud with one click.
                </div>
            </div>
            """)
            gr.HTML(
                "<p style='margin:8px 0 12px 0; font-size:0.95rem; line-height:1.6;'>"
                "Use the <strong>Local / Cloud</strong> toggle to switch between your local machine "
                "and a cloud provider. Switching pre-fills the fields — you only need "
                "to add your password. Click <strong>Test Connection</strong> to verify before chatting."
                "</p>"
            )

            # --- Mode toggle + cloud provider selector ---
            with gr.Row(variant="panel"):
                db_mode = gr.Radio(
                    choices=["Local", "Cloud"],
                    value="Local",
                    label="Database location",
                    info="Local = databases on this machine. Cloud = hosted provider.",
                    scale=1,
                    min_width=0,
                )
                cloud_provider = gr.Dropdown(
                    choices=list(_CLOUD_PROVIDER_HINTS.keys()),
                    value="Railway",
                    label="Cloud provider",
                    info="Select your provider to see setup instructions.",
                    scale=2,
                    min_width=0,
                    visible=False,
                )

            # provider_hint is kept as a hidden component so the toggle
            # wiring (_toggle_outputs) stays intact — just never shown.
            provider_hint = gr.Textbox(
                value="",
                interactive=False,
                visible=False,
            )

            gr.HTML('<div style="border-top:1px solid #E2E8F0; margin:12px 0;"></div>')

            # --- MySQL + PostgreSQL fields side by side ---
            with gr.Row(elem_id="db-fields-row", variant="panel"):
                # MySQL
                with gr.Column(scale=1, min_width=0):
                    gr.Markdown("#### MySQL")
                    mysql_host = gr.Textbox(
                        value=_LOCAL_DEFAULTS["mysql"]["host"],
                        label="Host",
                        placeholder="localhost",
                    )
                    mysql_port = gr.Textbox(
                        value=_LOCAL_DEFAULTS["mysql"]["port"],
                        label="Port",
                        placeholder="3306",
                    )
                    mysql_user = gr.Textbox(
                        value=_LOCAL_DEFAULTS["mysql"]["user"],
                        label="Username",
                        placeholder="root",
                    )
                    mysql_password = gr.Textbox(
                        value="",
                        label="Password",
                        type="password",
                        placeholder="Enter your MySQL password",
                    )
                    mysql_db = gr.Textbox(
                        value=_LOCAL_DEFAULTS["mysql"]["db"],
                        label="Database name",
                        placeholder="pet_store_scm",
                    )
                    mysql_test_btn = gr.Button(
                        "Test MySQL Connection", variant="secondary"
                    )
                    mysql_status = gr.Textbox(
                        label="MySQL status",
                        interactive=False,
                        lines=4,
                        value="Not tested yet.",
                    )

                # PostgreSQL
                with gr.Column(scale=1, min_width=0):
                    gr.Markdown("#### PostgreSQL")
                    pg_host = gr.Textbox(
                        value=_LOCAL_DEFAULTS["pg"]["host"],
                        label="Host",
                        placeholder="localhost",
                    )
                    pg_port = gr.Textbox(
                        value=_LOCAL_DEFAULTS["pg"]["port"],
                        label="Port",
                        placeholder="5432",
                    )
                    pg_user = gr.Textbox(
                        value=_LOCAL_DEFAULTS["pg"]["user"],
                        label="Username",
                        placeholder="postgres",
                    )
                    pg_password = gr.Textbox(
                        value="",
                        label="Password",
                        type="password",
                        placeholder="Enter your PostgreSQL password",
                    )
                    pg_db = gr.Textbox(
                        value=_LOCAL_DEFAULTS["pg"]["db"],
                        label="Database name",
                        placeholder="pet_store_scm",
                    )
                    pg_test_btn = gr.Button(
                        "Test PostgreSQL Connection", variant="secondary"
                    )
                    pg_status = gr.Textbox(
                        label="PostgreSQL status",
                        interactive=False,
                        lines=4,
                        value="Not tested yet.",
                    )

            # All outputs that the toggle updates (11 values)
            _toggle_outputs = [
                mysql_host,
                mysql_port,
                mysql_user,
                mysql_db,
                pg_host,
                pg_port,
                pg_user,
                pg_db,
                mysql_status,
                pg_status,
                provider_hint,
            ]

            def _on_mode_change_ui(mode, provider):
                vals = _on_mode_change(mode, provider)
                # Show/hide the cloud provider dropdown and hint box
                show_cloud = mode == "Cloud"
                return (
                    gr.update(value=vals[0]),  # mysql_host
                    gr.update(value=vals[1]),  # mysql_port
                    gr.update(value=vals[2]),  # mysql_user
                    gr.update(value=vals[3]),  # mysql_db
                    gr.update(value=vals[4]),  # pg_host
                    gr.update(value=vals[5]),  # pg_port
                    gr.update(value=vals[6]),  # pg_user
                    gr.update(value=vals[7]),  # pg_db
                    gr.update(value=vals[8]),  # mysql_status
                    gr.update(value=vals[9]),  # pg_status
                    gr.update(value=vals[10]),  # provider_hint text
                )

            def _on_mode_visibility(mode):
                show = mode == "Cloud"
                return gr.update(visible=show)

            # When mode radio changes: update field values + show/hide cloud controls
            db_mode.change(
                _on_mode_visibility,
                inputs=[db_mode],
                outputs=[cloud_provider],
            )
            db_mode.change(
                _on_mode_change_ui,
                inputs=[db_mode, cloud_provider],
                outputs=_toggle_outputs,
            )

            # When cloud provider changes: update hint text and port defaults
            cloud_provider.change(
                _on_mode_change_ui,
                inputs=[db_mode, cloud_provider],
                outputs=_toggle_outputs,
            )

            # Wire test buttons
            mysql_test_btn.click(
                fn=lambda h, p, u, pw, d: _run_test_connection("mysql", h, p, u, pw, d),
                inputs=[mysql_host, mysql_port, mysql_user, mysql_password, mysql_db],
                outputs=[mysql_status],
            )
            pg_test_btn.click(
                fn=lambda h, p, u, pw, d: _run_test_connection(
                    "postgres", h, p, u, pw, d
                ),
                inputs=[pg_host, pg_port, pg_user, pg_password, pg_db],
                outputs=[pg_status],
            )

        # Toggle handler for the DB settings panel
        def _toggle_db(is_open):
            new_open = not is_open
            return (
                new_open,
                gr.update(visible=new_open),
                gr.update(value="Hide" if new_open else "Show"),
            )

        _db_toggle_btn.click(
            _toggle_db,
            inputs=[_db_open],
            outputs=[_db_open, _db_panel, _db_toggle_btn],
        )

        # Tabs
        with gr.Tabs():
            build_assistant_tab(
                provider_dd,
                model_dd,
                api_key_box,
                key_status_box,
                mysql_host,
                mysql_port,
                mysql_user,
                mysql_password,
                mysql_db,
                pg_host,
                pg_port,
                pg_user,
                pg_password,
                pg_db,
            )
            inv_cat_dd, inv_outs, inv_update = build_inventory_tab()
            build_analytics_tab()
            build_forecast_tab()
            build_mlops_tab(
                mysql_host,
                mysql_port,
                mysql_user,
                mysql_password,
                mysql_db,
                pg_host,
                pg_port,
                pg_user,
                pg_password,
                pg_db,
            )

        # Note: demo.load() was removed — it caused a JS error on page load
        # that broke all tab navigation. Click "Refresh Dashboard" instead.

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:16px; margin-top:20px;
                    border-top: 1.5px solid #DADCE0; background:#F8F9FA; border-radius:12px;">
            <span style="color:#5F6368; font-size:13px; font-weight:500;">
                Pet Store SCM &nbsp;&bull;&nbsp;
                ReAct Agent &rarr; MCP Server &rarr; MySQL + PostgreSQL &nbsp;&bull;&nbsp;
                50 MCP Tools &nbsp;&bull;&nbsp; 47,515 rows &times; 65 SKUs &times; Pet Store India
            </span>
        </div>
        """)

    return demo


# Entry point — build UI immediately so the app is accessible right away.
# CatBoost training runs in a background daemon thread (≈60 s on CPU).
# Forecasts use the statistical fallback until training finishes, then
# automatically switch to CatBoost on the next request — no restart needed.
import threading as _threading

# ── Load saved models BEFORE building the UI so the forecast tab works immediately.
# This is a fast operation (reads pkl files, ~1-3s) — no need to defer to background.
# If no cached model exists, _init_ml() will silently return and the user can
# train from the MLOps Monitor tab.
try:
    from forecasting.ml_forecast import (
        load_models as _load_saved_models,
        get_metrics as _get_saved_metrics,
    )

    if _load_saved_models():
        _ml_metrics = _get_saved_metrics()
        _ml_ready = True
        import logging as _logging

        _logging.getLogger(__name__).info(
            f"[app] Model pre-loaded: {_ml_metrics.get('engine')} "
            f"MAPE={_ml_metrics.get('mape', '?')}% "
            f"trained {_ml_metrics.get('trained_at', '?')}"
        )
except Exception as _e:
    pass  # No cached model — user can train from MLOps Monitor

demo = build_ui()

# Background thread only needed if no cached model was found
# (to attempt loading from the configured data source on first run)
if not _ml_ready:
    _ml_thread = _threading.Thread(target=_init_ml, daemon=True)
    _ml_thread.start()

if __name__ == "__main__":
    import socket as _socket

    def _free_port(start: int = 7860, end: int = 7880) -> int:
        """Return the first free TCP port in [start, end)."""
        for p in range(start, end):
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return start  # fall back to default and let Gradio report the error

    port = _free_port(int(os.getenv("PORT", 7860)))
    print(f"\n  App is running at:  http://localhost:{port}\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=_build_theme(),
        css=MINIMAL_CSS,
        js=_FORCE_LIGHT_JS,
        allowed_paths=[str(BASE_DIR)],
    )
