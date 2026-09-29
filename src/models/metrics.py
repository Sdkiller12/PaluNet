"""Métriques de classification binaire et calibration du seuil (B4, B6, B7).

numpy pur : utilisable dans les tests, l'évaluation et l'audit de dérive sans
dépendance à TensorFlow ni scikit-learn. Convention : y_true = 1 pour la classe
positive (Parasitized), scores = probabilité de la classe positive, décision
positive si score >= seuil.
"""
from __future__ import annotations

import numpy as np


def _rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    """AUC ROC via la statistique de Mann-Whitney (gère les ex-aequo)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(y_true, y_pred) -> dict[str, int]:
    y, p = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return {
        "tp": int((y & p).sum()),
        "fn": int((y & ~p).sum()),
        "fp": int((~y & p).sum()),
        "tn": int((~y & ~p).sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_metrics(y_true, scores, threshold: float) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    cm = confusion(y_true, s >= threshold)
    tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {
        "n": int(len(s)),
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, len(s)),
        "recall": recall,
        "precision": precision,
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "auc": roc_auc(y_true, s),
        "confusion_matrix": cm,
    }


def choose_threshold(y_true, scores, target_recall: float) -> dict:
    """B6 — seuil maximisant la précision sous contrainte rappel >= target_recall.

    Parcourt tous les seuils candidats (scores distincts). À précision égale, le
    seuil le plus élevé est retenu. Si la cible est inatteignable (impossible en
    pratique : seuil minimal => rappel 1), lève ValueError.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0:
        raise ValueError("Aucun exemple positif : calibration impossible")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    # Ne garder que la dernière position de chaque valeur distincte (seuil = cette valeur)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    thr, tp, fp = s_sorted[last], tp[last], fp[last]
    recall = tp / y.sum()
    precision = tp / (tp + fp)
    ok = recall >= target_recall
    if not ok.any():
        raise ValueError(f"Rappel cible {target_recall} inatteignable")
    idx = np.flatnonzero(ok)
    best = idx[np.lexsort((-thr[idx], -precision[idx]))[0]]
    return {
        "threshold": float(thr[best]),
        "recall": float(recall[best]),
        "precision": float(precision[best]),
        "target_recall": float(target_recall),
    }


def check_against_baseline(metrics: dict, baseline: dict, tolerance: float) -> list[str]:
    """E4 — liste des régressions (métrique < baseline - tolérance)."""
    failures = []
    for key in ("accuracy", "recall", "auc"):
        if key in baseline and metrics[key] < baseline[key] - tolerance:
            failures.append(f"{key}: {metrics[key]:.4f} < baseline {baseline[key]:.4f} - {tolerance}")
    return failures


def check_acceptance(metrics: dict, acceptance: dict) -> list[str]:
    """Critères d'acceptation absolus (1.3)."""
    return [
        f"{k}: {metrics[k]:.4f} < cible {acceptance[k]}"
        for k in ("accuracy", "recall", "auc")
        if metrics[k] < acceptance[k]
    ]
