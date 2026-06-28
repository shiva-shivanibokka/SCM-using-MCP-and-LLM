"""Database engine helper.

The app reads from PostgreSQL (Neon) when DATABASE_URL is set, and falls back
to the bundled CSVs otherwise — so local dev and tests work with no database,
and the deployed app uses the real warehouse.
"""
from __future__ import annotations

import os
from functools import lru_cache


def _normalize(url: str) -> str:
    # SQLAlchemy wants the psycopg2 driver spelled out; Neon/Heroku hand out
    # "postgres://" or "postgresql://" which we rewrite to "postgresql+psycopg2://".
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


@lru_cache(maxsize=1)
def get_engine():
    """Return a cached SQLAlchemy engine, or None if DATABASE_URL isn't set."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return None
    from sqlalchemy import create_engine

    return create_engine(_normalize(url), pool_pre_ping=True, pool_recycle=300)


def database_enabled() -> bool:
    return get_engine() is not None
