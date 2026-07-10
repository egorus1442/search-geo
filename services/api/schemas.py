"""Pydantic v2 схемы для API запросов и ответов."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Localize ──────────────────────────────────────────────────────────────────

class LocalizeResponse(BaseModel):
    task_id: str
    status: Literal["completed", "pending", "failed"]
    processing_time_ms: int | None = None
    candidates: list[CandidateItem] = []
    error: str | None = None


class CandidateItem(BaseModel):
    rank: int
    patch_id: int
    center_lat: float
    center_lon: float
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    inlier_count: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    thumbnail_url: str | None = None


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[lon_min, lat_min, lon_max, lat_max]",
        examples=[[35.0, 50.0, 40.0, 55.0]],
    )
    date_from: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2023-06-01"])
    date_to: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2024-09-30"])
    cloud_cover_max: float = Field(default=20.0, ge=0, le=100)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        lon_min, lat_min, lon_max, lat_max = v
        if lon_min >= lon_max:
            raise ValueError("lon_min must be < lon_max")
        if lat_min >= lat_max:
            raise ValueError("lat_min must be < lat_max")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError("Longitude out of range")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError("Latitude out of range")
        return v


class TaskResponse(BaseModel):
    task_id: str
    status: str
    type: str
    result: Any = None
    error: str | None = None


# ── Index stats ───────────────────────────────────────────────────────────────

class IndexStatsResponse(BaseModel):
    faiss_ntotal: int
    faiss_dim: int
    faiss_n_lists: int
    faiss_n_probe: int
    faiss_is_trained: bool
    coarse_method: str = "unknown"
    vocab_size: int  # для VLAD — размерность coarse-вектора (dim), для BoVW — размер словаря
    patch_count_db: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, str]
