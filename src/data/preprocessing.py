"""Prétraitement UNIQUE partagé entraînement / inférence (A2, 3.2).

Volontairement sans dépendance TensorFlow : l'API l'utilise telle quelle, et le
pipeline d'entraînement met en cache la sortie de ces mêmes fonctions. Toute
modification ici change le comportement du modèle -> couverte par des tests de
non-régression.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG"}
DEFAULT_SIZE = 128


class InvalidImageError(ValueError):
    """Image illisible, corrompue ou de format non autorisé (-> HTTP 400)."""


@dataclass
class QualityReport:
    ok: bool
    issues: list[str] = field(default_factory=list)


def decode_image(data: bytes, allowed_formats: set[str] = ALLOWED_FORMATS) -> Image.Image:
    """Décode des octets en image PIL RGB en vérifiant le format RÉEL (pas l'extension)."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()  # détecte les fichiers tronqués / corrompus
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("Fichier image illisible ou corrompu") from exc
    if fmt not in allowed_formats:
        raise InvalidImageError(f"Format '{fmt}' non supporté (attendu : PNG ou JPEG)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (OSError, SyntaxError) as exc:
        raise InvalidImageError("Fichier image illisible ou corrompu") from exc
    return to_rgb(img)


def load_image_file(path: str | Path) -> Image.Image:
    with open(path, "rb") as f:
        return decode_image(f.read())


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB sur fond noir (fond natif des images NIH) pour les images avec transparence."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def preprocess(img: Image.Image, size: int = DEFAULT_SIZE) -> np.ndarray:
    """PIL RGB -> tableau float32 (size, size, 3) normalisé dans [0, 1]."""
    resized = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def preprocess_uint8(img: Image.Image, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Même transformation que `preprocess`, stockée en uint8 (cache d'entraînement)."""
    resized = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def uint8_to_model_input(arr: np.ndarray) -> np.ndarray:
    """Cache uint8 -> entrée modèle ; identique à `preprocess` (division par 255)."""
    return arr.astype(np.float32) / 255.0


def check_quality(img: Image.Image, min_side: int = 32, min_std: float = 0.01) -> QualityReport:
    """Contrôle qualité image (IQA) minimal : taille et contenu non uniforme."""
    issues: list[str] = []
    w, h = img.size
    if min(w, h) < min_side:
        issues.append(f"Image trop petite ({w}x{h} px, minimum {min_side} px)")
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    if arr.std() < min_std:
        issues.append("Image quasi uniforme (vide, noire ou saturée)")
    return QualityReport(ok=not issues, issues=issues)
