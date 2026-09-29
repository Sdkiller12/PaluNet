"""B1 — Entraînement reproductible (transfer learning MobileNetV2, 2 phases).

Usage :
    python -m src.data.prepare  (une fois)  puis
    python -m src.models.train [--limit 2000 --epochs1 1 --epochs2 1]   # smoke test

Le jeu de TEST n'est jamais chargé ici (A4) : l'early stopping ne voit que la validation.
Produit : models/keras/model_v<version>.keras, logs/training_v<version>_phase{1,2}.csv,
          ligne de synthèse dans logs/experiments.csv (E8).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import ROOT_DIR, label_of, load_config, positive_probability, resolve_path
from src.models.artifacts import ModelPaths
from src.models.metrics import classification_metrics
from src.models.model import build_model, set_backbone_trainable
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_split(processed: str, name: str, limit: int | None, seed: int):
    data = np.load(resolve_path(processed) / f"{name}.npz")
    x, y = data["x"], data["y"]
    if limit and limit < len(x):
        idx = np.random.default_rng(seed).choice(len(x), limit, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def augmentation(cfg: dict) -> keras.Sequential:
    a = cfg["augmentation"]
    aug = [
        layers.RandomRotation(a["rotation_deg"] / 360.0, fill_mode="constant"),
        layers.RandomTranslation(a["translation"], a["translation"], fill_mode="constant"),
        layers.RandomZoom((-a["zoom"], a["zoom"]), fill_mode="constant"),
    ]
    if a.get("horizontal_flip"):
        aug.insert(0, layers.RandomFlip("horizontal"))
    return keras.Sequential(aug, name="augmentation")


def make_dataset(x, y, batch_size: int, seed: int, augment: keras.Sequential | None = None):
    ds = tf.data.Dataset.from_tensor_slices((x, y.astype(np.float32)))
    if augment is not None:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    # Même normalisation que preprocessing.uint8_to_model_input
    ds = ds.batch(batch_size).map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b), num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        ds = ds.map(lambda a, b: (augment(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Recall(name="recall"),      # rappel de la classe d'indice 1
            keras.metrics.Precision(name="precision"),
        ],
    )


def callbacks(cfg: dict, log_path) -> list:
    es = cfg["training"]["early_stopping"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor=es["monitor"], patience=es["patience"], restore_best_weights=True),
        keras.callbacks.CSVLogger(str(log_path)),
    ]


def append_experiment(row: dict) -> None:
    path = ROOT_DIR / "logs" / "experiments.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    cfg = load_config()
    t = cfg["training"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=cfg["model"]["version"])
    parser.add_argument("--epochs1", type=int, default=t["phase1"]["epochs"])
    parser.add_argument("--epochs2", type=int, default=t["phase2"]["epochs"])
    parser.add_argument("--batch-size", type=int, default=t["batch_size"])
    parser.add_argument("--limit", type=int, help="Sous-échantillon (smoke test)")
    parser.add_argument("--deterministic", action="store_true", help="Opérations déterministes (plus lent)")
    args = parser.parse_args()

    seed = cfg["seed"]
    keras.utils.set_random_seed(seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    paths = ModelPaths(args.version)
    processed = cfg["data"]["processed_dir"]

    x_tr, y_tr = load_split(processed, "train", args.limit, seed)
    x_va, y_va = load_split(processed, "val", args.limit and max(args.limit // 4, 64), seed)
    log.info("Train %s | Val %s", x_tr.shape, x_va.shape)
    train_ds = make_dataset(x_tr, y_tr, args.batch_size, seed, augmentation(cfg))
    val_ds = make_dataset(x_va, y_va, args.batch_size, seed)

    model = build_model(cfg)
    start = time.time()

    log.info("Phase 1 : extraction de features (backbone gelé)")
    compile_model(model, t["phase1"]["learning_rate"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs1,
              callbacks=callbacks(cfg, paths.training_log.with_name(f"training_v{args.version}_phase1.csv")))

    log.info("Phase 2 : fine-tuning des %d dernières couches", t["phase2"]["unfreeze_last_layers"])
    set_backbone_trainable(model, t["phase2"]["unfreeze_last_layers"])
    compile_model(model, t["phase2"]["learning_rate"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs2,
              callbacks=callbacks(cfg, paths.training_log.with_name(f"training_v{args.version}_phase2.csv")))

    paths.keras.parent.mkdir(parents=True, exist_ok=True)
    model.save(paths.keras)
    log.info("Modèle sauvegardé : %s", paths.keras)

    # B4 — métriques de validation au seuil neutre (indicatif ; le seuil réel est calibré ensuite)
    p = positive_probability(model.predict(val_ds, verbose=0).reshape(-1), cfg["class_map"], cfg["positive_class"])
    pos = label_of(cfg["positive_class"], cfg["class_map"])
    val_metrics = classification_metrics(y_va == pos, p, 0.5)
    log.info("Validation (seuil 0.5, avant calibration) : %s", json.dumps(val_metrics))

    append_experiment({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": args.version,
        "train_size": len(x_tr), "val_size": len(x_va),
        "epochs1": args.epochs1, "epochs2": args.epochs2, "batch_size": args.batch_size,
        "lr1": t["phase1"]["learning_rate"], "lr2": t["phase2"]["learning_rate"],
        "val_accuracy@0.5": round(val_metrics["accuracy"], 4),
        "val_recall@0.5": round(val_metrics["recall"], 4),
        "val_auc": round(val_metrics["auc"], 4),
        "duration_s": round(time.time() - start),
    })


if __name__ == "__main__":
    main()
