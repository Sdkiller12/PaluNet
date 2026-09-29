"""Configuration de l'API par variables d'environnement (3.6)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.config import load_config, resolve_path
from src.models.artifacts import ModelPaths


def _opt_float(name: str) -> float | None:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model_path: str
    model_version: str | None = None          # surcharge la fiche modèle si fourni
    decision_threshold: float | None = None   # surcharge la fiche modèle si fourni
    mock_model: bool = False
    api_keys: tuple[str, ...] = ()            # vide => authentification désactivée (dev local)
    rate_limit: str = "60/minute"
    rate_limit_storage: str = "memory://"     # ex. redis://... pour partager entre instances
    max_upload_mb: float = 5.0
    max_batch_size: int = 32
    log_level: str = "INFO"
    prediction_log_file: str | None = None
    log_retention_days: int = 365
    intra_op_threads: int | None = None
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    serve_web_app: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> "Settings":
        cfg = load_config()
        default_model = str(ModelPaths(cfg["model"]["version"]).onnx)
        threads = os.environ.get("INTRA_OP_THREADS")
        return cls(
            model_path=str(resolve_path(os.environ.get("MODEL_PATH") or default_model)),
            model_version=os.environ.get("MODEL_VERSION") or None,
            decision_threshold=_opt_float("DECISION_THRESHOLD"),
            mock_model=_bool("MOCK_MODEL"),
            api_keys=tuple(k.strip() for k in os.environ.get("API_KEY", "").split(",") if k.strip()),
            rate_limit=os.environ.get("RATE_LIMIT", "60/minute"),
            rate_limit_storage=os.environ.get("RATE_LIMIT_STORAGE", "memory://"),
            max_upload_mb=float(os.environ.get("MAX_UPLOAD_MB", 5)),
            max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", 32)),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            prediction_log_file=os.environ.get("PREDICTION_LOG_FILE") or None,
            log_retention_days=int(os.environ.get("LOG_RETENTION_DAYS", 365)),
            intra_op_threads=int(threads) if threads else None,
            cors_origins=tuple(o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()),
            serve_web_app=_bool("SERVE_WEB_APP", True),
        )
