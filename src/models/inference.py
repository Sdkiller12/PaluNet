"""Moteur d'inférence ONNX Runtime — partagé par l'API, la calibration et l'évaluation.

Garantit que le seuil est calibré et les métriques mesurées sur l'artefact
réellement déployé (le .onnx), pas sur le modèle Keras d'entraînement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import label_of, positive_probability


@dataclass
class RawOutput:
    sigmoid: np.ndarray          # (N,) = P(classe d'indice 1)
    cam: np.ndarray | None       # (N, h, w) Grad-CAM brut vers la classe 1, ou None


class Predictor(Protocol):
    has_cam: bool

    def run(self, batch: np.ndarray, with_cam: bool = False) -> RawOutput: ...


class OnnxPredictor:
    def __init__(self, model_path: str | Path | None = None, intra_op_threads: int | None = None, session=None):
        if session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if intra_op_threads:
                opts.intra_op_num_threads = intra_op_threads
            session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        outputs = session.get_outputs()
        self.prob_name = self._pick(outputs, "probability", rank=2)
        if self.prob_name is None:
            raise ValueError("Sortie 'probability' introuvable dans le modèle ONNX")
        self.cam_name = self._pick(outputs, "cam", rank=3)
        self.has_cam = self.cam_name is not None

    @staticmethod
    def _pick(outputs, name: str, rank: int) -> str | None:
        for o in outputs:
            if o.name == name or o.name.startswith(name + ":"):
                return o.name
        for o in outputs:
            if len(o.shape) == rank:
                return o.name
        return None

    def run(self, batch: np.ndarray, with_cam: bool = False) -> RawOutput:
        names = [self.prob_name] + ([self.cam_name] if with_cam and self.has_cam else [])
        res = self.session.run(names, {self.input_name: batch.astype(np.float32, copy=False)})
        return RawOutput(sigmoid=np.asarray(res[0]).reshape(-1), cam=res[1] if len(res) > 1 else None)

    def predict_uint8(self, x_uint8: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Sigmoïde sur un cache uint8 (N, H, W, 3), par lots (évaluation hors ligne)."""
        out = [
            self.run(x_uint8[i : i + batch_size].astype(np.float32) / 255.0).sigmoid
            for i in range(0, len(x_uint8), batch_size)
        ]
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class MockPredictor:
    """Modèle factice déterministe (Sprint 1 : front/back développés contre le contrat figé).

    Heuristique grossière : les parasites apparaissent comme des taches violettes
    (bleu > rouge) alors que le cytoplasme est rose (rouge > bleu) ; le score
    augmente avec la part de pixels violets. NE PAS utiliser
    pour une quelconque décision.
    """

    has_cam = True

    def run(self, batch: np.ndarray, with_cam: bool = False) -> RawOutput:
        gray = batch.mean(axis=-1)
        cell = gray > 0.05                                   # ignore le fond noir
        dark = (batch[..., 2] > batch[..., 0]) & cell
        frac = dark.sum(axis=(1, 2)) / np.maximum(cell.sum(axis=(1, 2)), 1)
        sigmoid = 1.0 / (1.0 + np.exp(-(frac - 0.02) * 200.0))
        cam = None
        if with_cam:
            n, h, w = dark.shape
            cam = dark.reshape(n, 4, h // 4, 4, w // 4).mean(axis=(2, 4)) - 0.01
        return RawOutput(sigmoid=sigmoid.astype(np.float32), cam=cam)


@dataclass
class Decision:
    prediction: str
    confidence: float                 # probabilité de la classe PRÉDITE
    probability_parasitized: float    # probabilité de la classe positive, toujours
    predicted_index: int


def decide(sigmoid: float, threshold: float, class_map: dict[int, str], positive_class: str) -> Decision:
    """Applique le seuil calibré (jamais 0.5 implicite) à la probabilité de la classe positive."""
    p_pos = float(positive_probability(float(sigmoid), class_map, positive_class))
    pos_idx = label_of(positive_class, class_map)
    neg_idx = 1 - pos_idx
    if p_pos >= threshold:
        return Decision(class_map[pos_idx], p_pos, p_pos, pos_idx)
    return Decision(class_map[neg_idx], 1.0 - p_pos, p_pos, neg_idx)
