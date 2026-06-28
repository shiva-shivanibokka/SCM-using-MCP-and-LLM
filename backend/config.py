from __future__ import annotations
from dataclasses import dataclass
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
