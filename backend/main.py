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

app.include_router(suppliers_routes.router)
app.include_router(stores_routes.router)
app.include_router(analytics_routes.router)

from backend.api.routes import chat as chat_routes  # noqa: E402

app.include_router(chat_routes.router)
