"""FastAPI app for FULL-mode scanning."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ads.attribution import BackendInputName
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
    backend: BackendInputName = "toy"
    allow_fallback: bool = False
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    max_score_floor: float = Field(default=0.05, ge=0.0, le=5.0)
    redact: bool = True
    redaction_snippet_chars: int = Field(default=96, ge=8, le=512)


class PredictionPayload(BaseModel):
    """Detector prediction payload."""

    groundedness_score: float
    predicted_label: int
    confidence: float
    abstain_flag: bool


class ThresholdsPayload(BaseModel):
    """Threshold bundle used for scan inference."""

    decision_threshold: float
    score_threshold: float
    max_score_floor: float


class TopInfluentialItem(BaseModel):
    """Single influential training sample."""

    train_id: str
    score: float
    text: str
    text_hash: str | None = None
    meta: dict[str, str | int | float | bool] = Field(default_factory=dict)


class EvidenceHighlightPayload(BaseModel):
    """High-level evidence highlight entry."""

    train_id: str
    score: float
    score_share: float
    snippet: str
    text_hash: str


class EvidenceSummaryPayload(BaseModel):
    """Aggregated evidence summary for explainability."""

    item_count: int
    top1_share: float
    top5_share: float
    top10_share: float
    highlights: list[EvidenceHighlightPayload] = Field(default_factory=list)


class RedactionPayload(BaseModel):
    """Redaction metadata for top influential texts."""

    enabled: bool
    snippet_chars: int
    hash_algorithm: Literal["sha256"] = "sha256"
    full_text_included: bool


class ScanResponse(BaseModel):
    """Response payload for /scan endpoint."""

    prompt: str
    answer: str
    requested_detector: Literal["threshold", "logistic"]
    detector: Literal["threshold", "logistic"]
    fallback_reason: str | None = None
    features: dict[str, float | bool | int]
    prediction: PredictionPayload
    thresholds: ThresholdsPayload
    top_influential: list[TopInfluentialItem]
    evidence_summary: EvidenceSummaryPayload
    redaction: RedactionPayload


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/runtime/cache/clear")
def clear_cache() -> dict[str, str]:
    """Clear backend/model runtime caches used by FULL mode."""
    clear_runtime_caches()
    return {"status": "cleared"}


@app.exception_handler(ValueError)
def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
    """Translate scan value errors into explicit 4xx payloads."""
    message = str(exc)
    code = "MODEL_MISSING" if "logistic model missing:" in message else "INVALID_REQUEST"
    return JSONResponse(
        status_code=400,
        content={
            "error": message,
            "code": code,
        },
    )


@app.exception_handler(NotImplementedError)
def handle_not_implemented(_: Request, exc: NotImplementedError) -> JSONResponse:
    """Translate backend capability gaps into readable 4xx payloads."""
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "code": "BACKEND_UNAVAILABLE",
        },
    )


@app.post("/scan", response_model=ScanResponse)
def scan(payload: ScanRequest) -> ScanResponse:
    """Run ADS scan for one prompt/answer pair."""
    result = scan_sample(
        prompt=payload.prompt,
        answer=payload.answer,
        top_k=payload.top_k,
        backend_name=payload.backend,
        method=payload.method,
        allow_fallback=payload.allow_fallback,
        decision_threshold=payload.decision_threshold,
        score_threshold=payload.score_threshold,
        max_score_floor=payload.max_score_floor,
        redact=payload.redact,
        redaction_snippet_chars=payload.redaction_snippet_chars,
    )
    return ScanResponse(**result)
