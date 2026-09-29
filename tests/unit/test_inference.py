from types import SimpleNamespace

import numpy as np
import pytest

from src.config import label_of, parse_class_map, positive_probability, validate_class_map
from src.models.explain import class_activation, heatmap_base64, render_heatmap
from src.models.inference import MockPredictor, OnnxPredictor, decide
from src.data.preprocessing import preprocess
from tests.conftest import make_cell

CLASS_MAP = {0: "Uninfected", 1: "Parasitized"}
INVERTED = {0: "Parasitized", 1: "Uninfected"}


# --- B9 : encodage de classe ------------------------------------------------------
def test_config_class_map_is_frozen(cfg):
    assert cfg["class_map"] == CLASS_MAP
    assert cfg["positive_class"] == "Parasitized"


def test_class_map_validation():
    assert parse_class_map({"0": "A", "1": "B"}) == {0: "A", 1: "B"}
    with pytest.raises(ValueError):
        validate_class_map({0: "A", 1: "A"}, "A")
    with pytest.raises(ValueError):
        validate_class_map({0: "A", 2: "B"}, "A")
    with pytest.raises(ValueError):
        validate_class_map({0: "A", 1: "B"}, "C")
    with pytest.raises(KeyError):
        label_of("C", {0: "A", 1: "B"})


def test_positive_probability_respects_encoding():
    assert positive_probability(0.8, CLASS_MAP, "Parasitized") == pytest.approx(0.8)
    assert positive_probability(0.8, INVERTED, "Parasitized") == pytest.approx(0.2)


# --- Décision et désambiguïsation de `confidence` (3.3) -----------------------------
@pytest.mark.parametrize("class_map,sigmoid", [(CLASS_MAP, 0.9), (INVERTED, 0.1)])
def test_decide_positive(class_map, sigmoid):
    d = decide(sigmoid, 0.6, class_map, "Parasitized")
    assert d.prediction == "Parasitized"
    assert d.confidence == pytest.approx(0.9)
    assert d.probability_parasitized == pytest.approx(0.9)
    assert class_map[d.predicted_index] == "Parasitized"


def test_decide_negative_confidence_refers_to_predicted_class():
    d = decide(0.3, 0.6, CLASS_MAP, "Parasitized")
    assert d.prediction == "Uninfected"
    assert d.confidence == pytest.approx(0.7)
    assert d.probability_parasitized == pytest.approx(0.3)


def test_decide_uses_calibrated_threshold_not_half():
    # 0.4 < 0.5 mais >= seuil calibré 0.35 : doit être positif
    assert decide(0.4, 0.35, CLASS_MAP, "Parasitized").prediction == "Parasitized"


# --- ONNX Runtime (session factice) -------------------------------------------------
class FakeSession:
    def __init__(self, names_ranks):
        self._outputs = [SimpleNamespace(name=n, shape=[None] * r) for n, r in names_ranks]

    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def get_outputs(self):
        return self._outputs

    def run(self, names, feeds):
        n = feeds["image"].shape[0]
        table = {"probability": np.full((n, 1), 0.75, np.float32), "cam": np.ones((n, 4, 4), np.float32)}
        return [table[k.split(":")[0]] for k in names]


def test_onnx_predictor_with_and_without_cam():
    p = OnnxPredictor(session=FakeSession([("probability", 2), ("cam", 3)]))
    out = p.run(np.zeros((2, 128, 128, 3), np.float32), with_cam=True)
    assert out.sigmoid.shape == (2,) and out.cam.shape == (2, 4, 4)
    assert p.run(np.zeros((1, 128, 128, 3), np.float32)).cam is None
    assert p.predict_uint8(np.zeros((5, 128, 128, 3), np.uint8), batch_size=2).shape == (5,)


def test_onnx_predictor_without_cam_output():
    p = OnnxPredictor(session=FakeSession([("probability:0", 2)]))
    assert not p.has_cam
    assert p.run(np.zeros((1, 8, 8, 3), np.float32), with_cam=True).cam is None


def test_onnx_predictor_requires_probability_output():
    with pytest.raises(ValueError):
        OnnxPredictor(session=FakeSession([("cam", 3)]))


def test_mock_predictor_separates_synthetic_cells():
    batch = np.stack([preprocess(make_cell(True)), preprocess(make_cell(False))])
    out = MockPredictor().run(batch, with_cam=True)
    assert out.sigmoid[0] > 0.5 > out.sigmoid[1]
    assert out.cam.shape == (2, 4, 4)


# --- Grad-CAM -----------------------------------------------------------------------
def test_class_activation_sign_depends_on_predicted_class():
    cam = np.array([[1.0, -2.0], [0.5, 0.0]])
    np.testing.assert_allclose(class_activation(cam, 1), [[1.0, 0.0], [0.5, 0.0]])
    np.testing.assert_allclose(class_activation(cam, 0), [[0.0, 1.0], [0.0, 0.0]])
    assert class_activation(np.zeros((2, 2)), 1).max() == 0


def test_heatmap_rendering():
    img = render_heatmap(np.array([[0.0, 1.0], [0.5, 0.2]]), 64)
    assert img.size == (64, 64) and img.mode == "RGBA"
    assert isinstance(heatmap_base64(np.ones((4, 4)), 1, 32), str)
