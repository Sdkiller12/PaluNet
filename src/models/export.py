"""B5 — Export ONNX (+ quantification int8 si > max_size_mb) et fiche modèle JSON.

Usage :  python -m src.models.export [--version 1.0.0] [--force-quantize]

Vérifie l'équivalence Keras / ONNX Runtime sur un échantillon de validation.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow import keras

from src.config import load_config, resolve_path
from src.models.artifacts import ModelPaths, write_json
from src.models.inference import OnnxPredictor
from src.models.model import build_inference_model
from src.utils.logger import get_logger

log = get_logger(__name__)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize(src, dst) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)


def verify(keras_model: keras.Model, onnx_path, sample: np.ndarray, atol: float) -> float:
    expected_prob, expected_cam = keras_model.predict(sample, verbose=0)
    got = OnnxPredictor(onnx_path).run(sample, with_cam=True)
    diff = float(np.abs(expected_prob.reshape(-1) - got.sigmoid).max())
    cam_diff = float(np.abs(expected_cam - got.cam).max()) if got.cam is not None else float("nan")
    log.info("Écart max Keras/ONNX : probabilité %.2e, cam %.2e", diff, cam_diff)
    if diff > atol:
        raise SystemExit(f"Export invalide : écart {diff:.2e} > tolérance {atol}")
    return diff


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=cfg["model"]["version"])
    parser.add_argument("--force-quantize", action="store_true")
    args = parser.parse_args()

    paths = ModelPaths(args.version)
    size = cfg["preprocessing"]["image_size"]
    model = keras.models.load_model(paths.keras)
    inference_model = build_inference_model(model)

    spec = (tf.TensorSpec((None, size, size, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(inference_model, input_signature=spec, opset=cfg["export"]["opset"],
                               output_path=str(paths.onnx))

    val = np.load(resolve_path(cfg["data"]["processed_dir"]) / "val.npz")
    sample = val["x"][:64].astype(np.float32) / 255.0
    verify(inference_model, paths.onnx, sample, atol=1e-4)

    size_mb = paths.onnx.stat().st_size / 1e6
    quantized = False
    if size_mb > cfg["export"]["max_size_mb"] or args.force_quantize:
        log.info("Artefact %.1f Mo -> quantification int8", size_mb)
        fp32 = paths.onnx.with_name(paths.onnx.stem + "_fp32.onnx")
        paths.onnx.replace(fp32)
        quantize(fp32, paths.onnx)
        verify(inference_model, paths.onnx, sample, atol=0.05)
        size_mb, quantized = paths.onnx.stat().st_size / 1e6, True

    write_json(paths.metadata, {
        "model_version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_map": {str(k): v for k, v in cfg["class_map"].items()},
        "positive_class": cfg["positive_class"],
        "sigmoid_output": f"P(class_map[1]) = P({cfg['class_map'][1]})",
        "input": {"name": "image", "shape": [None, size, size, 3], "dtype": "float32",
                  "preprocessing": "RGB, resize bilinéaire, /255 -> [0,1] (src/data/preprocessing.py)"},
        "outputs": ["probability", "cam"],
        "size_mb": round(size_mb, 2),
        "quantized_int8": quantized,
        "sha256": sha256(paths.onnx),
        "decision_threshold": None,   # renseigné par calibrate_threshold.py (B6)
    })
    log.info("ONNX exporté : %s (%.1f Mo) — lancer ensuite la calibration", paths.onnx, size_mb)


if __name__ == "__main__":
    main()
