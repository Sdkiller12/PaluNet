"""Dépendances FastAPI : service modèle, authentification par clé API, rate limiting (C7)."""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.service import ModelService
from src.api.settings import Settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> ModelService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé")
    return service


def verify_api_key(request: Request, provided: str | None = Security(api_key_header)) -> None:
    keys = request.app.state.settings.api_keys
    if not keys:
        return  # authentification désactivée (API_KEY non défini : réseau local / dev uniquement)
    provided = provided or ""
    if not any(hmac.compare_digest(provided.encode(), k.encode()) for k in keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clé API manquante ou invalide (en-tête X-API-Key)")


def rate_limit_key(request: Request) -> str:
    """Quota par clé API (C7) ; par adresse IP si l'authentification est désactivée."""
    key = request.headers.get("X-API-Key")
    return f"key:{key}" if key else f"ip:{get_remote_address(request)}"


def build_limiter(settings: Settings) -> Limiter:
    return Limiter(key_func=rate_limit_key, storage_uri=settings.rate_limit_storage)
