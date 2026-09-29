import numpy as np
import pytest

from src.models.metrics import (
    check_acceptance,
    check_against_baseline,
    choose_threshold,
    classification_metrics,
    confusion,
    roc_auc,
)


def test_auc_perfect_random_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)  # nombreux ex-aequo
    pos, neg = s[y == 1], s[y == 0]
    expected = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert roc_auc(y, s) == pytest.approx(expected)


def test_auc_single_class_is_nan():
    assert np.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_confusion_and_metrics():
    y = [1, 1, 1, 0, 0]
    s = [0.9, 0.6, 0.2, 0.7, 0.1]
    assert confusion(y, np.array(s) >= 0.5) == {"tp": 2, "fn": 1, "fp": 1, "tn": 1}
    m = classification_metrics(y, s, 0.5)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_choose_threshold_meets_recall_target_with_best_precision():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    s = np.array([0.95, 0.9, 0.8, 0.4, 0.85, 0.3, 0.2, 0.1, 0.35, 0.33])
    res = choose_threshold(y, s, target_recall=0.8)
    assert res["recall"] >= 0.8
    # 0.40 : rappel 0.8, précision 4/5 ; 0.35 : rappel 1.0, précision 5/6 -> 0.35 retenu
    assert res["threshold"] == pytest.approx(0.35)
    assert res["precision"] == pytest.approx(5 / 6)
    m = classification_metrics(y, s, res["threshold"])
    assert m["recall"] == pytest.approx(res["recall"])
    full = choose_threshold(y, s, target_recall=1.0)
    assert classification_metrics(y, s, full["threshold"])["recall"] == 1.0


def test_choose_threshold_requires_positives():
    with pytest.raises(ValueError):
        choose_threshold([0, 0], [0.1, 0.2], 0.9)


def test_baseline_and_acceptance_checks():
    baseline = {"accuracy": 0.96, "recall": 0.97, "auc": 0.99}
    assert check_against_baseline({"accuracy": 0.956, "recall": 0.97, "auc": 0.99}, baseline, 0.005) == []
    fails = check_against_baseline({"accuracy": 0.95, "recall": 0.97, "auc": 0.99}, baseline, 0.005)
    assert len(fails) == 1 and fails[0].startswith("accuracy")
    acc = {"accuracy": 0.95, "recall": 0.96, "auc": 0.97}
    assert check_acceptance({"accuracy": 0.96, "recall": 0.95, "auc": 0.98}, acc) == ["recall: 0.9500 < cible 0.96"]
