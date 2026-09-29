"""Chemins et métadonnées des artefacts modèle versionnés (B8)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"
BASELINE_DIR = ROOT_DIR / "metrics_baseline"
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class ModelPaths:
    version: str

    def __post_init__(self):
        if not SEMVER.match(self.version):
            raise ValueError(f"model_version doit être au format semver X.Y.Z, reçu '{self.version}'")

    @property
    def keras(self) -> Path:
        return MODELS_DIR / "keras" / f"model_v{self.version}.keras"

    @property
    def onnx(self) -> Path:
        return MODELS_DIR / f"model_v{self.version}.onnx"

    @property
    def metadata(self) -> Path:
        return metadata_path_for(self.onnx)

    @property
    def training_log(self) -> Path:
        return ROOT_DIR / "logs" / f"training_v{self.version}.csv"

    @property
    def baseline(self) -> Path:
        return BASELINE_DIR / f"{self.version}.json"


def metadata_path_for(onnx_path: Path) -> Path:
    """Fiche modèle JSON associée à un .onnx (même nom, extension .json)."""
    return Path(onnx_path).with_suffix(".json")


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def update_metadata(path: Path, **fields) -> dict:
    data = read_json(path) if path.exists() else {}
    data.update(fields)
    write_json(path, data)
    return data
