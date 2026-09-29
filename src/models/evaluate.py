"""B7 / E4 / E5 — Évaluation de l'artefact ONNX avec le seuil calibré.

Usage :
    python -m src.models.evaluate                       # jeu de test (+ externe si présent)
    python -m src.models.evaluate --write-baseline      # fige metrics_baseline/<version>.json
    python -m src.models.evaluate --check-baseline      # non-régression (exit 1 si régression)
    python -m src.models.evaluate --images-dir audit/   # audit de dérive mensuel (E5) sur un
                                                        # dossier Parasitized/ + Uninfected/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import label_of, load_config, positive_probability, resolve_path
from src.data.preprocessing import InvalidImageError, load_image_file, preprocess_uint8
from src.data.split import list_samples
from src.models.artifacts import ModelPaths, read_json, write_json
from src.models.inference import OnnxPredictor
from src.models.metrics import check_acceptance, check_against_baseline, classification_metrics
from src.utils.logger import get_logger

log = get_logger(__name__)


def score_arrays(predictor: OnnxPredictor, x_uint8: np.ndarray, y: np.ndarray, cfg: dict):
    """-> (y_pos bool, p_pos float) orientés vers la classe positive."""
    p_pos = positive_probability(predictor.predict_uint8(x_uint8), cfg["class_map"], cfg["positive_class"])
    return y == label_of(cfg["positive_class"], cfg["class_map"]), p_pos


def load_npz(name: str, cfg: dict):
    path = resolve_path(cfg["data"]["processed_dir"]) / f"{name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["x"], d["y"]


def load_folder(folder: Path, cfg: dict):
    size = cfg["preprocessing"]["image_size"]
    xs, ys = [], []
    for s in list_samples(folder, cfg["class_map"]):
        try:
            xs.append(preprocess_uint8(load_image_file(s.path), size))
            ys.append(s.label)
        except InvalidImageError:
            log.warning("Image ignorée : %s", s.path)
    if not xs:
        raise SystemExit(f"Aucune image dans {folder} (attendu : sous-dossiers {list(cfg['class_map'].values())})")
    return np.stack(xs), np.asarray(ys)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=cfg["model"]["version"])
    parser.add_argument("--images-dir", help="Dossier étiqueté à évaluer (audit de dérive)")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--check-baseline", action="store_true")
    args = parser.parse_args()

    paths = ModelPaths(args.version)
    meta = read_json(paths.metadata)
    threshold = meta.get("decision_threshold")
    if threshold is None:
        raise SystemExit("Seuil non calibré : lancer `python -m src.models.calibrate_threshold` d'abord (B6)")
    predictor = OnnxPredictor(paths.onnx)
    acc = cfg["acceptance"]

    if args.images_dir:
        m = classification_metrics(*score_arrays(predictor, *load_folder(Path(args.images_dir), cfg), cfg), threshold)
        print(json.dumps(m, indent=2))
        if m["recall"] < acc["drift_alert_recall"]:
            log.error("ALERTE DÉRIVE : rappel %.4f < %.2f — déclencher le processus de réentraînement",
                      m["recall"], acc["drift_alert_recall"])
            sys.exit(2)
        return

    results = {}
    for name in ("test", "external"):
        data = load_npz(name, cfg)
        if data is not None and len(data[0]):
            results[name] = classification_metrics(*score_arrays(predictor, *data, cfg), threshold)
    if "test" not in results:
        raise SystemExit("data/processed/test.npz introuvable — lancer `python -m src.data.prepare`")
    print(json.dumps(results, indent=2))

    failures = check_acceptance(results["test"], acc)
    for f in failures:
        log.warning("Critère d'acceptation non atteint (test) — %s", f)

    if args.write_baseline:
        baseline = {k: round(results["test"][k], 4) for k in ("accuracy", "recall", "auc", "precision", "specificity")}
        write_json(paths.baseline, {
            "model_version": args.version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "decision_threshold": threshold,
            "tolerance": acc["tolerance"],
            "test": baseline,
            "test_size": results["test"]["n"],
            "external": results.get("external"),
            "acceptance_passed": not failures,
        })
        log.info("Baseline écrite : %s", paths.baseline)

    if args.check_baseline:
        regressions = check_against_baseline(results["test"], read_json(paths.baseline)["test"], acc["tolerance"])
        if regressions:
            log.error("RÉGRESSION : %s", regressions)
            sys.exit(1)
        log.info("Non-régression OK")


if __name__ == "__main__":
    main()
