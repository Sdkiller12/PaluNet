"""B6 — Calibration du seuil de décision sur le jeu de VALIDATION (jamais le test).

Usage :  python -m src.models.calibrate_threshold [--target-recall 0.98]

Choisit le seuil maximisant la précision sous contrainte rappel >= cible, sur
l'artefact ONNX exporté, puis l'écrit dans la fiche modèle (decision_threshold).
La variable d'environnement DECISION_THRESHOLD de l'API peut le surcharger.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from src.config import load_config
from src.models.artifacts import ModelPaths, update_metadata
from src.models.evaluate import load_npz, score_arrays
from src.models.inference import OnnxPredictor
from src.models.metrics import choose_threshold, classification_metrics
from src.utils.logger import get_logger

log = get_logger(__name__)


def pr_table(y_pos, p_pos, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    """Points de la courbe précision/rappel, pour la documentation du choix."""
    return [
        {k: round(v, 4) if isinstance(v, float) else v
         for k, v in classification_metrics(y_pos, p_pos, t).items()
         if k in ("threshold", "recall", "precision", "specificity")}
        for t in thresholds
    ]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=cfg["model"]["version"])
    parser.add_argument("--target-recall", type=float, default=cfg["calibration"]["target_recall"])
    args = parser.parse_args()

    paths = ModelPaths(args.version)
    data = load_npz("val", cfg)
    if data is None:
        raise SystemExit("data/processed/val.npz introuvable — lancer `python -m src.data.prepare`")
    y_pos, p_pos = score_arrays(OnnxPredictor(paths.onnx), *data, cfg)

    choice = choose_threshold(y_pos, p_pos, args.target_recall)
    # Arrondi VERS LE BAS : un seuil plus bas ne peut que conserver (ou augmenter) le rappel
    threshold = float(np.floor(choice["threshold"] * 1e4) / 1e4)
    val_metrics = classification_metrics(y_pos, p_pos, threshold)
    update_metadata(
        paths.metadata,
        decision_threshold=threshold,
        calibration={
            "dataset": "validation",
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "précision maximale sous contrainte rappel >= cible",
            "target_recall": args.target_recall,
            "val_recall": round(val_metrics["recall"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_auc": round(val_metrics["auc"], 4),
            "val_size": val_metrics["n"],
            "pr_curve": pr_table(y_pos, p_pos),
        },
    )
    log.info("Seuil calibré : %.4f (%s)", threshold, json.dumps(choice))


if __name__ == "__main__":
    main()
