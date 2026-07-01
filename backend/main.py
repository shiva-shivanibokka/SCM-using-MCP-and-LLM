"""FastAPI entrypoint for Petopia Intelligence Hub."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Petopia Intelligence Hub API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # Any localhost port (dev server + vite preview) and any Vercel deployment.
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/diagnostics")
def diagnostics():
    """Report the live data source so we can tell Postgres from CSV fallback.

    `database_configured` is whether DATABASE_URL is set; `database_live` is
    whether an actual `SELECT 1` succeeds right now. If configured is true but
    live is false, the app is silently serving the CSV fallback.
    """
    from backend.db import database_enabled, get_engine

    source = "csv"
    db_live = False
    detail = None
    engine = get_engine()
    if engine is not None:
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_live = True
            source = "postgres"
        except Exception as e:  # configured but unreachable -> CSV fallback
            detail = str(e)[:200]

    return {
        "data_source": source,
        "database_configured": database_enabled(),
        "database_live": db_live,
        "error": detail,
    }


from backend.api.routes import forecast as forecast_routes  # noqa: E402
from backend.api.routes import mlops as mlops_routes  # noqa: E402

app.include_router(forecast_routes.router)
app.include_router(mlops_routes.router)

from backend.api.routes import executive as executive_routes  # noqa: E402
from backend.api.routes import inventory as inventory_routes  # noqa: E402

app.include_router(executive_routes.router)
app.include_router(inventory_routes.router)

from backend.api.routes import suppliers as suppliers_routes  # noqa: E402
from backend.api.routes import stores as stores_routes  # noqa: E402
from backend.api.routes import analytics as analytics_routes  # noqa: E402
from backend.api.routes import recommendations as recommendations_routes  # noqa: E402
from backend.api.routes import intelligence as intelligence_routes  # noqa: E402

app.include_router(suppliers_routes.router)
app.include_router(stores_routes.router)
app.include_router(analytics_routes.router)
app.include_router(recommendations_routes.router)
app.include_router(intelligence_routes.router)

from backend.api.routes import chat as chat_routes  # noqa: E402

app.include_router(chat_routes.router)
