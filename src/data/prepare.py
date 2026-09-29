"""Construction des splits (A4) et du cache prétraité (A2) + jeu externe (A6).

Usage :  python -m src.data.prepare

Produit dans data/processed/ :
  - manifest.csv            : chemin, label, patient_id, split (traçabilité)
  - {train,val,test}.npz    : images uint8 (N,128,128,3) + labels, issues de
                              `preprocessing.preprocess_uint8` (même code que l'API)
  - external.npz            : jeu externe si data/external/ est rempli
  - split_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import ROOT_DIR, load_config, resolve_path
from src.data.preprocessing import InvalidImageError, load_image_file, preprocess_uint8
from src.data.split import (
    SPLITS,
    Sample,
    group_stratified_split,
    list_samples,
    split_summary,
    write_manifest,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_cache(samples: list[Sample], size: int, out: Path) -> int:
    images, labels, skipped = [], [], 0
    for i, s in enumerate(samples, 1):
        try:
            images.append(preprocess_uint8(load_image_file(s.path), size))
            labels.append(s.label)
        except InvalidImageError:
            skipped += 1
            log.warning("Image ignorée (illisible) : %s", s.path)
        if i % 5000 == 0:
            log.info("  %s : %d/%d", out.name, i, len(samples))
    x = np.stack(images) if images else np.zeros((0, size, size, 3), np.uint8)
    np.savez(out, x=x, y=np.asarray(labels, dtype=np.int64))
    return skipped


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=cfg["data"]["raw_dir"])
    parser.add_argument("--out-dir", default=cfg["data"]["processed_dir"])
    args = parser.parse_args()

    raw, out = resolve_path(args.raw_dir), resolve_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["preprocessing"]["image_size"]
    class_map = cfg["class_map"]

    samples = list_samples(raw, class_map)
    if not samples:
        raise SystemExit(f"Aucune image trouvée dans {raw} — lancer d'abord `python -m src.data.download`")
    samples = group_stratified_split(samples, cfg["data"]["split"], seed=cfg["seed"])
    write_manifest(samples, out / "manifest.csv", root=ROOT_DIR)
    summary = split_summary(samples)
    log.info("Répartition : %s", json.dumps(summary))

    for name in SPLITS:
        skipped = build_cache([s for s in samples if s.split == name], size, out / f"{name}.npz")
        summary[name]["skipped"] = skipped

    external = resolve_path(cfg["data"]["external_dir"])
    ext_samples = list_samples(external, class_map) if all((external / c).is_dir() for c in class_map.values()) else []
    if ext_samples:
        ext_samples = [Sample(s.path, s.label, s.patient_id, "external") for s in ext_samples]
        build_cache(ext_samples, size, out / "external.npz")
        summary["external"] = split_summary(ext_samples)["external"]
        if len(ext_samples) < 200:
            log.warning("Jeu externe : %d images (< 200 requis par A6)", len(ext_samples))
    else:
        log.warning("Jeu externe absent (%s) — requis par A6 avant la recette", external)

    (out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Cache prêt dans %s", out)


if __name__ == "__main__":
    main()
