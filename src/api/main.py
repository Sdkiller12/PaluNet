"""Application FastAPI — API d'inférence PaluNet (module C).

Lancement :  uvicorn src.api.main:app --port 8000
Mode mock :  MOCK_MODEL=true uvicorn src.api.main:app   (sans modèle entraîné)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.dependencies import build_limiter
from src.api.routes import health, predict
from src.api.service import ModelLoadError, ModelService
from src.api.settings import Settings
from src.config import ROOT_DIR
from src.utils.logger import configure_logging

log = logging.getLogger(__name__)
API_VERSION = "1.0.0"
DISCLAIMER = ("Outil d'aide à la décision — ne remplace pas un diagnostic médical certifié. "
              "Version de démonstration.")


class RequestMetrics:
    """Compteurs en mémoire, par instance (l'API reste stateless côté métier)."""

    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.latencies: deque[float] = deque(maxlen=window)
        self.status_counts: Counter[int] = Counter()

    def record(self, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.status_counts[status_code] += 1
            self.latencies.append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies)
            total = sum(self.status_counts.values())
            errors = sum(c for s, c in self.status_counts.items() if s >= 400)

        def pct(p: float):
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None

        return {"requests_total": total, "error_rate": round(errors / total, 4) if total else 0.0,
                "status_codes": {str(k): v for k, v in sorted(self.status_counts.items())},
                "latency_ms": {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "window": len(lat)}}


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, settings.prediction_log_file, settings.log_retention_days)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            try:
                app.state.service = ModelService.from_settings(settings)
            except ModelLoadError as exc:
                # L'instance reste vivante (liveness) mais non prête (readiness /health = 503)
                app.state.load_error = str(exc)
                log.error("Échec du chargement du modèle : %s", exc)
        if not settings.api_keys:
            log.warning("API_KEY non défini : authentification désactivée (réservé au réseau local)")
        yield

    app = FastAPI(title="PaluNet — Détection du paludisme", version=API_VERSION,
                  description=DISCLAIMER, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.state.metrics = RequestMetrics()

    limiter = build_limiter(settings)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Erreur interne")
            response = JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})
        if request.url.path.startswith("/v1/"):
            app.state.metrics.record(response.status_code, (time.perf_counter() - start) * 1000)
        return response

    app.include_router(health.router)
    app.include_router(predict.build_router(limiter, settings.rate_limit))

    web_dir = ROOT_DIR / "web_app"
    if settings.serve_web_app and web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web_app")
    return app


app = create_app()
