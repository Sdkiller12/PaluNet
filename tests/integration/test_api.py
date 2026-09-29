import base64
import json
import logging

import pytest

from src.api.service import MOCK_VERSION, ModelLoadError, ModelService
from src.api.settings import Settings
from src.utils.logger import PREDICTION_LOGGER
from tests.conftest import make_cell, make_client, to_bytes


def post(client, data: bytes, name="cell.png", mime="image/png", **kw):
    return client.post("/v1/predict", files={"file": (name, data, mime)}, **kw)


# --- /health ----------------------------------------------------------------------
def test_health_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "1.0.0"
    assert body["decision_threshold"] == 0.5
    assert body["class_map"] == {"0": "Uninfected", "1": "Parasitized"}
    assert client.get("/health/live").json() == {"status": "alive"}


def test_health_503_when_model_missing(tmp_path):
    settings = Settings(model_path=str(tmp_path / "absent.onnx"), serve_web_app=False)
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    with TestClient(create_app(settings)) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert "introuvable" in r.json()["detail"]
        assert post(c, to_bytes(make_cell(True))).status_code == 503
        assert c.get("/health/live").status_code == 200


# --- /v1/predict ------------------------------------------------------------------
def test_predict_contract(client, infected_png):
    r = post(client, infected_png)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"prediction", "confidence", "probability_parasitized", "model_version",
                         "processing_time_ms", "heatmap_base64"}
    assert body["prediction"] == "Parasitized"
    assert body["confidence"] == body["probability_parasitized"]
    assert body["heatmap_base64"] is None


def test_confidence_refers_to_predicted_class(client, healthy_png):
    body = post(client, healthy_png).json()
    assert body["prediction"] == "Uninfected"
    assert body["confidence"] == pytest.approx(1 - body["probability_parasitized"], abs=1e-4)


@pytest.mark.parametrize("how", ["form", "query"])
def test_predict_with_heatmap(client, infected_png, how):
    if how == "form":
        r = client.post("/v1/predict", files={"file": ("c.png", infected_png, "image/png")},
                        data={"include_heatmap": "true"})
    else:
        r = post(client, infected_png, params={"include_heatmap": "true"})
    heatmap = base64.b64decode(r.json()["heatmap_base64"])
    assert heatmap.startswith(b"\x89PNG")


def test_jpeg_accepted(client):
    assert post(client, to_bytes(make_cell(True), "JPEG"), "c.jpg", "image/jpeg").status_code == 200


def test_quality_warning_returned(client):
    from PIL import Image

    body = post(client, to_bytes(Image.new("RGB", (16, 16)))).json()
    assert len(body["quality_warnings"]) == 2


@pytest.mark.parametrize("data,expected", [
    (b"definitely not an image", 400),
    (b"", 400),
])
def test_invalid_files_rejected(client, data, expected):
    r = post(client, data)
    assert r.status_code == expected
    assert r.json()["detail"]


def test_unsupported_format_rejected(client):
    r = post(client, to_bytes(make_cell(True), "GIF"), "c.gif", "image/gif")
    assert r.status_code == 400
    assert "non supporté" in r.json()["detail"]


def test_too_large_rejected(cfg):
    with make_client(cfg, max_upload_mb=0.001) as c:
        assert post(c, to_bytes(make_cell(True))).status_code == 413


def test_missing_file_is_422(client):
    assert client.post("/v1/predict").status_code == 422


# --- C7 : clé API et rate limiting --------------------------------------------------
def test_api_key_required_when_configured(cfg, infected_png):
    with make_client(cfg, api_keys=("secret-1", "secret-2")) as c:
        assert post(c, infected_png).status_code == 401
        assert post(c, infected_png, headers={"X-API-Key": "wrong"}).status_code == 401
        assert post(c, infected_png, headers={"X-API-Key": "secret-2"}).status_code == 200
        assert c.get("/health").status_code == 200  # la sonde reste publique


def test_rate_limit_per_key(cfg, infected_png):
    with make_client(cfg, api_keys=("a", "b"), rate_limit="3/minute") as c:
        codes = [post(c, infected_png, headers={"X-API-Key": "a"}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert post(c, infected_png, headers={"X-API-Key": "b"}).status_code == 200


# --- /v1/predict/batch --------------------------------------------------------------
def test_batch_partial_failure(client, infected_png, healthy_png):
    files = [
        ("files", ("a.png", infected_png, "image/png")),
        ("files", ("broken.png", b"xxx", "image/png")),
        ("files", ("b.png", healthy_png, "image/png")),
    ]
    r = client.post("/v1/predict/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert [x["filename"] for x in body["results"]] == ["a.png", "b.png"]
    assert [x["prediction"] for x in body["results"]] == ["Parasitized", "Uninfected"]
    assert body["errors"] == [{"filename": "broken.png", "status_code": 400,
                               "detail": "Fichier image illisible ou corrompu"}]


def test_batch_limit(client, infected_png):
    files = [("files", (f"{i}.png", infected_png, "image/png")) for i in range(33)]
    assert client.post("/v1/predict/batch", files=files).status_code == 400


# --- C5 : logs anonymisés, /metrics -------------------------------------------------
def test_prediction_log_is_anonymised(client, infected_png, caplog):
    with caplog.at_level(logging.INFO, logger=PREDICTION_LOGGER):
        post(client, infected_png, "C100P61ThinF_IMG_20150918_cell_1.png")
    records = [r for r in caplog.records if r.name == PREDICTION_LOGGER]
    assert len(records) == 1
    rec = records[0].__dict__
    assert rec["prediction"] == "Parasitized" and "prediction_id" in rec
    dumped = json.dumps({k: str(v) for k, v in rec.items()})
    assert "C100P61" not in dumped and "testclient" not in dumped


def test_metrics_endpoint(client, infected_png):
    post(client, infected_png)
    post(client, b"bad")
    m = client.get("/metrics").json()
    assert m["requests_total"] == 2
    assert m["error_rate"] == 0.5
    assert m["latency_ms"]["p95"] is not None
    assert m["model_version"] == "1.0.0"


# --- Chargement du modèle (B8/B9) ---------------------------------------------------
def write_model(tmp_path, meta: dict):
    onnx = tmp_path / "model_v1.0.0.onnx"
    onnx.write_bytes(b"fake")
    (tmp_path / "model_v1.0.0.json").write_text(json.dumps(meta), encoding="utf-8")
    return onnx


GOOD_META = {"model_version": "1.0.0", "decision_threshold": 0.42,
             "class_map": {"0": "Uninfected", "1": "Parasitized"}}


def test_service_loads_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.service.OnnxPredictor", lambda *a, **k: type("P", (), {"has_cam": True})())
    svc = ModelService.from_settings(Settings(model_path=str(write_model(tmp_path, GOOD_META))))
    assert (svc.model_version, svc.threshold) == ("1.0.0", 0.42)
    svc = ModelService.from_settings(Settings(model_path=str(write_model(tmp_path, GOOD_META)),
                                              decision_threshold=0.3, model_version="1.0.1"))
    assert (svc.model_version, svc.threshold) == ("1.0.1", 0.3)


@pytest.mark.parametrize("meta,msg", [
    ({**GOOD_META, "class_map": {"0": "Parasitized", "1": "Uninfected"}}, "class_map"),
    ({**GOOD_META, "decision_threshold": None}, "seuil"),
    ({**GOOD_META, "decision_threshold": 1.5}, "invalide"),
])
def test_service_refuses_inconsistent_model(tmp_path, monkeypatch, meta, msg):
    monkeypatch.setattr("src.api.service.OnnxPredictor", lambda *a, **k: type("P", (), {"has_cam": True})())
    with pytest.raises(ModelLoadError, match=msg):
        ModelService.from_settings(Settings(model_path=str(write_model(tmp_path, meta))))


def test_service_requires_metadata(tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    with pytest.raises(ModelLoadError, match="Fiche"):
        ModelService.from_settings(Settings(model_path=str(onnx)))


def test_mock_mode():
    svc = ModelService.from_settings(Settings(model_path="none", mock_model=True))
    assert svc.model_version == MOCK_VERSION


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "k1, k2")
    monkeypatch.setenv("DECISION_THRESHOLD", "0.37")
    monkeypatch.setenv("MOCK_MODEL", "true")
    monkeypatch.setenv("INTRA_OP_THREADS", "2")
    s = Settings.from_env()
    assert s.api_keys == ("k1", "k2")
    assert s.decision_threshold == 0.37 and s.mock_model and s.intra_op_threads == 2
    assert s.model_path.endswith("model_v1.0.0.onnx")
    assert s.max_upload_bytes == 5 * 1024 * 1024
