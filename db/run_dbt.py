"""Run dbt against the same Neon database the app uses.

dbt-postgres wants discrete connection fields, but the app stores a single
DATABASE_URL. This wrapper parses DATABASE_URL into the DBT_* env vars that
dbt/profiles.yml reads, then invokes dbt with whatever args you pass:

    python db/run_dbt.py build     # run all models + tests
    python db/run_dbt.py test      # just the data-quality tests
    python db/run_dbt.py run       # just build the models
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BASE / "backend" / ".env")
load_dotenv(BASE / ".env")


def main() -> None:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        print("ERROR: DATABASE_URL not set. Add it to .env first.")
        sys.exit(1)

    # Normalize to a plain postgresql:// URL so urlparse gives clean parts.
    for prefix in ("postgresql+psycopg2://", "postgres://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
    p = urlparse(url)

    os.environ["DBT_HOST"] = p.hostname or ""
    os.environ["DBT_USER"] = unquote(p.username or "")
    os.environ["DBT_PASSWORD"] = unquote(p.password or "")
    os.environ["DBT_PORT"] = str(p.port or 5432)
    os.environ["DBT_DBNAME"] = (p.path or "/").lstrip("/")
    os.environ.setdefault("DBT_SCHEMA", "analytics")

    project = str(BASE / "dbt")
    args = sys.argv[1:] or ["build"]
    cmd = ["dbt", *args, "--project-dir", project, "--profiles-dir", project]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
