"""Grad-CAM (C9 / D6) : rendu de la carte de chaleur à partir de la sortie `cam` de l'ONNX.

Le calcul Grad-CAM lui-même est intégré au graphe exporté (voir
`model.build_inference_model`) ; ce module ne fait que du post-traitement numpy.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Points de contrôle d'une palette type "jet" (bleu -> cyan -> jaune -> rouge)
_STOPS = np.array([0.0, 0.35, 0.65, 1.0])
_COLORS = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]], dtype=np.float32)


def class_activation(cam_raw: np.ndarray, predicted_index: int) -> np.ndarray:
    """Carte (h, w) dans [0, 1] des zones ayant motivé la classe prédite."""
    signed = cam_raw if predicted_index == 1 else -cam_raw
    cam = np.maximum(signed, 0.0).astype(np.float32)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def render_heatmap(activation: np.ndarray, size: int, max_alpha: float = 0.6) -> Image.Image:
    """Carte [0,1] -> image RGBA (size x size) à superposer sur l'image d'origine."""
    up = Image.fromarray(activation.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR)
    a = np.clip(np.asarray(up), 0.0, 1.0)
    rgb = np.stack([np.interp(a, _STOPS, _COLORS[:, c]) for c in range(3)], axis=-1)
    alpha = (a * max_alpha * 255.0)[..., None]
    return Image.fromarray(np.concatenate([rgb, alpha], axis=-1).astype(np.uint8), mode="RGBA")


def heatmap_base64(cam_raw: np.ndarray, predicted_index: int, size: int) -> str:
    img = render_heatmap(class_activation(cam_raw, predicted_index), size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
