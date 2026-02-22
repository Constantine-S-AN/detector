"""FastAPI app for FULL-mode scanning."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ads.service import clear_runtime_caches, scan_sample

app = FastAPI(title="Attribution Density Scanner API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    """Request payload for /scan endpoint."""

    prompt: str
    answer: str
    method: Literal["threshold", "logistic"] = "logistic"
    top_k: int = Field(default=20, ge=1, le=200)
    backend: Literal["toy", "trak", "cea", "dda"] = "toy"


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/runtime/cache/clear")
def clear_cache() -> dict[str, str]:
    """Clear backend/model runtime caches used by FULL mode."""
    clear_runtime_caches()
    return {"status": "cleared"}


@app.post("/scan")
def scan(payload: ScanRequest) -> dict[str, object]:
    """Run ADS scan for one prompt/answer pair."""
    return scan_sample(
        prompt=payload.prompt,
        answer=payload.answer,
        top_k=payload.top_k,
        backend_name=payload.backend,
        method=payload.method,
    )
