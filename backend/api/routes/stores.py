from __future__ import annotations
from fastapi import APIRouter

from backend.data_access import load_stores

router = APIRouter(prefix="/api/stores", tags=["stores"])


@router.get("/grid")
def grid():
    df = load_stores()
    return {"stores": df.to_dict(orient="records")}
