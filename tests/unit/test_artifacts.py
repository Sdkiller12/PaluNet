import pytest

from src.models.artifacts import (
    BASELINE_DIR,
    MODELS_DIR,
    ModelPaths,
    metadata_path_for,
    read_json,
    update_metadata,
    write_json,
)
from src.config import ROOT_DIR


def test_model_paths_follow_versioning_convention():
    p = ModelPaths("1.2.3")
    assert p.keras == MODELS_DIR / "keras" / "model_v1.2.3.keras"
    assert p.onnx == MODELS_DIR / "model_v1.2.3.onnx"
    assert p.metadata == MODELS_DIR / "model_v1.2.3.json"
    assert p.training_log == ROOT_DIR / "logs" / "training_v1.2.3.csv"
    assert p.baseline == BASELINE_DIR / "1.2.3.json"


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "1.0.0-rc1", ""])
def test_model_paths_require_semver(version):
    with pytest.raises(ValueError, match="semver"):
        ModelPaths(version)


def test_metadata_path_for_replaces_extension(tmp_path):
    assert metadata_path_for(tmp_path / "m.onnx") == tmp_path / "m.json"


def test_json_roundtrip_creates_parents_and_keeps_unicode(tmp_path):
    path = tmp_path / "a" / "b" / "fiche.json"
    write_json(path, {"classe": "parasitée", "seuil": 0.32})
    assert read_json(path) == {"classe": "parasitée", "seuil": 0.32}
    text = path.read_text(encoding="utf-8")
    assert "parasitée" in text and text.endswith("\n")


def test_update_metadata_merges_and_creates(tmp_path):
    path = tmp_path / "fiche.json"
    assert update_metadata(path, a=1) == {"a": 1}
    assert update_metadata(path, b=2, a=3) == {"a": 3, "b": 2}
    assert read_json(path) == {"a": 3, "b": 2}
