"""Contrat d'API (3.3) — figé ; toute rupture passe par un nouveau préfixe /v2."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(..., examples=["Parasitized"])
    confidence: float = Field(..., ge=0, le=1, description=(
        "Probabilité associée à la classe PRÉDITE (champ `prediction`), pas à une classe fixe."))
    probability_parasitized: float = Field(..., ge=0, le=1, description=(
        "Probabilité que la cellule soit parasitée, quelle que soit la classe prédite."))
    model_version: str = Field(..., examples=["1.0.0"])
    processing_time_ms: int
    heatmap_base64: str | None = Field(None, description=(
        "PNG RGBA (Grad-CAM) encodé en base64, à superposer à l'image ; rempli si include_heatmap=true."))
    quality_warnings: list[str] = Field(default_factory=list, description=(
        "Avertissements du contrôle qualité image (la prédiction est tout de même fournie)."))


class BatchItem(PredictionResponse):
    filename: str


class BatchError(BaseModel):
    filename: str
    status_code: int
    detail: str


class BatchResponse(BaseModel):
    results: list[BatchItem]
    errors: list[BatchError]
    model_version: str
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
    decision_threshold: float | None = None
    class_map: dict[str, str] | None = None
    heatmap_available: bool | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
