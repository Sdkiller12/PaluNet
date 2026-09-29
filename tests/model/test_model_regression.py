"""E1/E4 — Non-régression du modèle réel. Ignoré tant que le modèle et les données n'existent pas."""
import numpy as np
import pytest

from src.config import label_of, load_config, resolve_path
from src.models.artifacts import ModelPaths, read_json
from src.models.metrics import check_against_baseline, classification_metrics

cfg = load_config()
paths = ModelPaths(cfg["model"]["version"])
test_npz = resolve_path(cfg["data"]["processed_dir"]) / "test.npz"

pytestmark = pytest.mark.skipif(
    not (paths.onnx.exists() and paths.metadata.exists() and paths.baseline.exists() and test_npz.exists()),
    reason="Modèle, baseline ou jeu de test absent (lancer le pipeline d'entraînement)",
)


@pytest.fixture(scope="module")
def scored():
    from src.models.evaluate import score_arrays
    from src.models.inference import OnnxPredictor

    data = np.load(test_npz)
    return score_arrays(OnnxPredictor(paths.onnx), data["x"], data["y"], cfg)


def test_no_regression_against_baseline(scored):
    threshold = read_json(paths.metadata)["decision_threshold"]
    metrics = classification_metrics(*scored, threshold)
    baseline = read_json(paths.baseline)
    assert check_against_baseline(metrics, baseline["test"], cfg["acceptance"]["tolerance"]) == []


def test_no_silent_class_inversion(scored):
    """Les cellules étiquetées parasitées doivent avoir un score moyen nettement plus élevé."""
    y_pos, p_pos = scored
    assert p_pos[y_pos].mean() > 0.8 and p_pos[~y_pos].mean() < 0.2


def test_model_size_under_limit():
    assert paths.onnx.stat().st_size / 1e6 < cfg["export"]["max_size_mb"]


def test_metadata_matches_config():
    meta = read_json(paths.metadata)
    assert {int(k): v for k, v in meta["class_map"].items()} == cfg["class_map"]
    assert label_of(cfg["positive_class"], cfg["class_map"]) in (0, 1)
    assert 0 < meta["decision_threshold"] < 1
