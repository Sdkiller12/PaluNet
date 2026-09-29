"""POST /v1/predict et /v1/predict/batch (C1, C2, C4, C9)."""

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from slowapi import Limiter

from src.api.dependencies import get_service, verify_api_key
from src.api.schemas import BatchResponse, ErrorResponse, PredictionResponse
from src.api.service import ModelService, PreparedImage
from src.data.preprocessing import InvalidImageError

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Format non supporté ou image corrompue"},
    401: {"model": ErrorResponse, "description": "Clé API manquante ou invalide"},
    413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
    429: {"description": "Limite de requêtes dépassée"},
    503: {"model": ErrorResponse, "description": "Modèle non chargé"},
}


def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Fichier trop volumineux (max {max_bytes // (1024 * 1024)} Mo)")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Fichier vide")
    return data


def load_upload(upload: UploadFile, service: ModelService, max_bytes: int) -> PreparedImage:
    data = read_limited(upload, max_bytes)
    try:
        return service.prepare(data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["prédiction"], dependencies=[Depends(verify_api_key)],
                       responses=ERROR_RESPONSES)

    @router.post("/predict", response_model=PredictionResponse)
    @limiter.limit(rate_limit)
    def predict(
        request: Request,
        file: UploadFile = File(..., description="Image PNG ou JPEG d'une cellule unique (max 5 Mo)"),
        include_heatmap: bool = Form(False),
        include_heatmap_query: bool = Query(False, alias="include_heatmap"),
        service: ModelService = Depends(get_service),
    ):
        image = load_upload(file, service, request.app.state.settings.max_upload_bytes)
        return service.predict([image], include_heatmap or include_heatmap_query)[0]

    @router.post("/predict/batch", response_model=BatchResponse)
    @limiter.limit(rate_limit)
    def predict_batch(
        request: Request,
        files: list[UploadFile] = File(..., description="Jusqu'à 32 images PNG/JPEG"),
        service: ModelService = Depends(get_service),
    ):
        settings = request.app.state.settings
        if len(files) > settings.max_batch_size:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"Trop d'images ({len(files)}), maximum {settings.max_batch_size} par requête")
        start = time.perf_counter()
        prepared, names, errors = [], [], []
        for f in files:
            try:
                prepared.append(load_upload(f, service, settings.max_upload_bytes))
                names.append(f.filename or "")
            except HTTPException as exc:
                errors.append({"filename": f.filename or "", "status_code": exc.status_code, "detail": exc.detail})
        # Une seule inférence pour tout le lot : pas de dégradation proportionnelle
        results = [{"filename": n, **r} for n, r in zip(names, service.predict(prepared, endpoint="batch"))]
        return {
            "results": results,
            "errors": errors,
            "model_version": service.model_version,
            "processing_time_ms": int(round((time.perf_counter() - start) * 1000)),
        }

    return router
