"""GET /health (readiness : 200 seulement si le modèle est chargé), /health/live (liveness), /metrics."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse

router = APIRouter(tags=["santé"])


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True,
            responses={503: {"model": HealthResponse}})
def health(request: Request):
    service = getattr(request.app.state, "service", None)
    if service is None:
        return JSONResponse(status_code=503, content={
            "status": "unavailable", "detail": getattr(request.app.state, "load_error", "Modèle non chargé")})
    return service.health()


@router.get("/health/live")
def live():
    return {"status": "alive"}


@router.get("/metrics")
def metrics(request: Request):
    """Métriques d'observabilité de l'instance (latence, erreurs, version, seuil)."""
    service = getattr(request.app.state, "service", None)
    snapshot = request.app.state.metrics.snapshot()
    snapshot.update({
        "model_version": service.model_version if service else None,
        "decision_threshold": service.threshold if service else None,
    })
    return snapshot
