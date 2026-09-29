"""Chargement de la configuration centrale et helpers d'encodage de classe (B9)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["class_map"] = parse_class_map(cfg["class_map"])
    validate_class_map(cfg["class_map"], cfg["positive_class"])
    return cfg


def parse_class_map(raw: dict) -> dict[int, str]:
    """Normalise les clés en int (YAML/JSON peuvent les fournir en str)."""
    return {int(k): str(v) for k, v in raw.items()}


def validate_class_map(class_map: dict[int, str], positive_class: str) -> None:
    if sorted(class_map) != [0, 1]:
        raise ValueError(f"class_map doit avoir exactement les clés 0 et 1, reçu {class_map}")
    if len(set(class_map.values())) != 2:
        raise ValueError(f"class_map contient des classes dupliquées : {class_map}")
    if positive_class not in class_map.values():
        raise ValueError(f"positive_class '{positive_class}' absente de class_map {class_map}")


def label_of(class_name: str, class_map: dict[int, str]) -> int:
    """Nom de classe -> indice entier selon class_map."""
    for idx, name in class_map.items():
        if name == class_name:
            return idx
    raise KeyError(f"Classe inconnue '{class_name}' (class_map={class_map})")


def positive_probability(sigmoid_out, class_map: dict[int, str], positive_class: str):
    """Sortie sigmoïde (= P(classe d'indice 1)) -> probabilité de la classe positive.

    Seul endroit où l'orientation de la sortie est interprétée : une inversion de
    class_map dans la config reste donc cohérente de bout en bout.
    """
    return sigmoid_out if label_of(positive_class, class_map) == 1 else 1.0 - sigmoid_out


def resolve_path(p: str | os.PathLike) -> Path:
    """Chemin relatif -> relatif à la racine du projet."""
    path = Path(p)
    return path if path.is_absolute() else ROOT_DIR / path
