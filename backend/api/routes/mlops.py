from __future__ import annotations
from fastapi import APIRouter

from backend.forecasting.registry import get_registry

router = APIRouter(prefix="/api/mlops", tags=["mlops"])


@router.get("/registry")
def registry():
    return get_registry()
