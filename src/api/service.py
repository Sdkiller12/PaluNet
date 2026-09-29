"""Service de prédiction : chargement du modèle versionné, validation des images, décision."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.api.settings import Settings
from src.config import load_config, parse_class_map
from src.data.preprocessing import check_quality, decode_image, preprocess
from src.models.artifacts import metadata_path_for, read_json
from src.models.explain import heatmap_base64
from src.models.inference import MockPredictor, OnnxPredictor, Predictor, decide
from src.utils.logger import PREDICTION_LOGGER

log = logging.getLogger(__name__)
prediction_log = logging.getLogger(PREDICTION_LOGGER)

MOCK_VERSION = "0.0.0-mock"
MOCK_THRESHOLD = 0.5  # seuil NON calibré, réservé au mode mock (jamais servi en réel)


class ModelLoadError(RuntimeError):
    pass


@dataclass
class PreparedImage:
    array: np.ndarray            # (H, W, 3) float32 [0, 1]
    quality_warnings: list[str]


class ModelService:
    def __init__(self, predictor: Predictor, model_version: str, threshold: float,
                 class_map: dict[int, str], positive_class: str, image_size: int, quality_cfg: dict):
        if not 0.0 < threshold < 1.0:
            raise ModelLoadError(f"decision_threshold invalide : {threshold}")
        self.predictor = predictor
        self.model_version = model_version
        self.threshold = threshold
        self.class_map = class_map
        self.positive_class = positive_class
        self.image_size = image_size
        self.quality_cfg = quality_cfg

    # --- Chargement -------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelService":
        cfg = load_config()
        common = dict(class_map=cfg["class_map"], positive_class=cfg["positive_class"],
                      image_size=cfg["preprocessing"]["image_size"],
                      quality_cfg={"min_side": cfg["preprocessing"]["min_side_px"],
                                   "min_std": cfg["preprocessing"]["min_std"]})
        if settings.mock_model:
            log.warning("MODE MOCK : prédictions factices, ne pas utiliser pour une décision")
            threshold = settings.decision_threshold or MOCK_THRESHOLD
            return cls(MockPredictor(), settings.model_version or MOCK_VERSION, threshold, **common)

        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise ModelLoadError(f"Modèle introuvable : {model_path}")
        meta_path = metadata_path_for(model_path)
        if not meta_path.exists():
            raise ModelLoadError(f"Fiche modèle introuvable : {meta_path}")
        meta = read_json(meta_path)

        # B9 — refuse de servir un modèle dont l'encodage diffère de la config
        model_class_map = parse_class_map(meta.get("class_map", {}))
        if model_class_map != cfg["class_map"]:
            raise ModelLoadError(f"class_map du modèle {model_class_map} != config {cfg['class_map']}")

        threshold = settings.decision_threshold if settings.decision_threshold is not None \
            else meta.get("decision_threshold")
        if threshold is None:
            raise ModelLoadError("Aucun seuil calibré (decision_threshold) : lancer calibrate_threshold (B6)")
        version = settings.model_version or meta.get("model_version")
        predictor = OnnxPredictor(model_path, intra_op_threads=settings.intra_op_threads)
        log.info("Modèle chargé", extra={"model_version": version, "decision_threshold": threshold,
                                         "model_path": str(model_path)})
        return cls(predictor, version, float(threshold), **common)

    @property
    def has_heatmap(self) -> bool:
        return self.predictor.has_cam

    # --- Prédiction -------------------------------------------------------------
    def prepare(self, data: bytes) -> PreparedImage:
        """Lève InvalidImageError si le fichier n'est pas une image PNG/JPEG valide."""
        img = decode_image(data)
        report = check_quality(img, **self.quality_cfg)
        return PreparedImage(preprocess(img, self.image_size), report.issues)

    def predict(self, images: list[PreparedImage], include_heatmap: bool = False,
                endpoint: str = "predict") -> list[dict]:
        if not images:
            return []
        start = time.perf_counter()
        batch = np.stack([im.array for im in images])
        raw = self.predictor.run(batch, with_cam=include_heatmap)
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        results = []
        for i, im in enumerate(images):
            d = decide(float(raw.sigmoid[i]), self.threshold, self.class_map, self.positive_class)
            heatmap = heatmap_base64(raw.cam[i], d.predicted_index, self.image_size) \
                if include_heatmap and raw.cam is not None else None
            result = {
                "prediction": d.prediction,
                "confidence": round(d.confidence, 4),
                "probability_parasitized": round(d.probability_parasitized, 4),
                "model_version": self.model_version,
                "processing_time_ms": elapsed_ms,
                "heatmap_base64": heatmap,
                "quality_warnings": im.quality_warnings,
            }
            self._log(result, endpoint)
            results.append(result)
        return results

    def _log(self, result: dict, endpoint: str) -> None:
        # C5 — aucune donnée patient : ni nom de fichier (les noms NIH contiennent
        # l'identifiant patient), ni image, ni IP. Un identifiant aléatoire par prédiction.
        prediction_log.info("prediction", extra={
            "prediction_id": uuid.uuid4().hex,
            "endpoint": endpoint,
            "prediction": result["prediction"],
            "probability_parasitized": result["probability_parasitized"],
            "decision_threshold": self.threshold,
            "model_version": self.model_version,
            "processing_time_ms": result["processing_time_ms"],
            "quality_warnings": len(result["quality_warnings"]),
        })

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_version": self.model_version,
            "decision_threshold": self.threshold,
            "class_map": {str(k): v for k, v in self.class_map.items()},
            "heatmap_available": self.has_heatmap,
        }
