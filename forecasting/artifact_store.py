"""Durable model-weight persistence in Postgres (Neon).

HuggingFace Space container storage is ephemeral: anything a fine-tune writes to
forecasting/.model_cache is wiped on the next rebuild/restart, so the model
silently reverts to an untrained state. This module zips the local model cache
and stores it as a BYTEA blob in Neon (which IS durable), and restores it on
boot when the local cache is missing.

Self-contained: builds its own engine from DATABASE_URL, and degrades to a no-op
(returns False) when there's no database or no sqlalchemy — so local/offline use
is unaffected. Storing model bytes next to the model_registry metadata keeps the
whole MLOps story in one durable place.
"""
from __future__ import annotations

import io
import logging
import os
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_TABLE = "model_artifacts"
# Keep only the most recent N versions per model (rollback history without
# unbounded growth). Blobs are small, so a handful is plenty.
_KEEP_LAST = 5


def _engine():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return None
    try:
        from sqlalchemy import create_engine

        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        return create_engine(url, pool_pre_ping=True, pool_recycle=300)
    except Exception as e:  # sqlalchemy/psycopg2 missing, bad URL, etc.
        logger.info("artifact_store: no engine (%s)", e)
        return None


def _ensure(conn) -> None:
    from sqlalchemy import text

    conn.execute(text(
        f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
        " id serial PRIMARY KEY,"
        " model_name text NOT NULL,"
        " created_at timestamptz DEFAULT now(),"
        " payload bytea NOT NULL)"
    ))


def _zip_dir(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(path.glob("*")):
            if f.is_file():
                z.write(f, f.name)
    return buf.getvalue()


def _unzip(data: bytes, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(path)


def save_cache(cache_dir, model_name: str = "catboost") -> bool:
    """Zip the model cache and store it durably in Neon. Best-effort."""
    p = Path(cache_dir)
    if not p.exists() or not any(p.glob("*")):
        return False
    eng = _engine()
    if eng is None:
        return False
    try:
        from sqlalchemy import text

        payload = _zip_dir(p)
        with eng.begin() as conn:
            _ensure(conn)
            conn.execute(
                text(f"INSERT INTO {_TABLE} (model_name, payload) VALUES (:n, :p)"),
                {"n": model_name, "p": payload},
            )
            # Retention: keep only the most recent _KEEP_LAST versions.
            conn.execute(
                text(
                    f"DELETE FROM {_TABLE} WHERE model_name = :n AND id NOT IN ("
                    f"  SELECT id FROM {_TABLE} WHERE model_name = :n"
                    f"  ORDER BY created_at DESC LIMIT :k)"
                ),
                {"n": model_name, "k": _KEEP_LAST},
            )
        logger.info("artifact_store: saved %s weights to Neon (%d bytes)", model_name, len(payload))
        return True
    except Exception as e:
        logger.warning("artifact_store: save failed (%s)", e)
        return False


def restore_cache(cache_dir, model_name: str = "catboost") -> bool:
    """Restore the most recent durable copy of the model cache from Neon."""
    eng = _engine()
    if eng is None:
        return False
    try:
        from sqlalchemy import text

        with eng.begin() as conn:
            _ensure(conn)
            row = conn.execute(
                text(f"SELECT payload FROM {_TABLE} WHERE model_name = :n "
                     "ORDER BY created_at DESC LIMIT 1"),
                {"n": model_name},
            ).fetchone()
        if not row:
            return False
        _unzip(bytes(row[0]), Path(cache_dir))
        logger.info("artifact_store: restored %s weights from Neon", model_name)
        return True
    except Exception as e:
        logger.warning("artifact_store: restore failed (%s)", e)
        return False
