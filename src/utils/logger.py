"""Logs JSON structurés (observabilité) + journal de prédictions anonymisé (C5)."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PREDICTION_LOGGER = "palunet.predictions"
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None, prediction_log_file: str | None = None,
                      retention_days: int = 365) -> None:
    """Configure la racine en JSON sur stdout.

    Si `prediction_log_file` est fourni, le journal des prédictions est aussi écrit
    dans un fichier à rotation quotidienne conservant `retention_days` fichiers :
    la purge au-delà de 12 mois est donc automatique (exigence de rétention).
    """
    root = logging.getLogger()
    root.setLevel((level or os.environ.get("LOG_LEVEL", "INFO")).upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consoles Windows (cp1252)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    pred_logger = logging.getLogger(PREDICTION_LOGGER)
    for h in list(pred_logger.handlers):
        pred_logger.removeHandler(h)
        h.close()
    if prediction_log_file:
        Path(prediction_log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler(
            prediction_log_file, when="midnight", backupCount=retention_days, encoding="utf-8", utc=True
        )
        fh.setFormatter(JsonFormatter())
        pred_logger.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        configure_logging()
    return logger
