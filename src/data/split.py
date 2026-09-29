"""A4 — Split train/val/test reproductible, stratifié et groupé PAR PATIENT.

Le dataset NIH contient de nombreuses cellules par patient (~200 patients).
Un split au niveau image ferait apparaître un même patient dans le train et le
test, gonflant artificiellement les métriques. On répartit donc des patients
entiers entre les splits, en équilibrant le nombre d'images de chaque classe.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    patient_id: str
    split: str = ""


def patient_id_from_filename(name: str) -> str:
    """'C100P61ThinF_IMG_20150918_144104_cell_162.png' -> 'C100P61ThinF'."""
    stem = Path(name).name
    return stem.split("_IMG")[0] if "_IMG" in stem else stem.rsplit("_", 1)[0]


def list_samples(root: Path, class_map: dict[int, str]) -> list[Sample]:
    samples = []
    for label, cls in sorted(class_map.items()):
        for p in sorted((root / cls).glob("*.png")):
            samples.append(Sample(path=str(p), label=label, patient_id=patient_id_from_filename(p.name)))
    return samples


def group_stratified_split(
    samples: list[Sample], fractions: dict[str, float], seed: int
) -> list[Sample]:
    """Affectation gloutonne des patients au split le plus en déficit (par classe).

    Déterministe pour un seed donné. Garantit qu'aucun patient n'apparaît dans
    deux splits.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"Les fractions doivent sommer à 1 : {fractions}")
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    for s in samples:
        counts[s.patient_id][s.label] += 1

    patients = sorted(counts)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    # Les gros patients d'abord : l'algorithme glouton équilibre mieux.
    patients.sort(key=lambda p: -counts[p].sum())

    totals = sum(counts.values())
    names = list(fractions)
    target = np.array([[fractions[n] * t for t in totals] for n in names])  # (splits, classes)
    current = np.zeros_like(target)
    assignment: dict[str, str] = {}
    for p in patients:
        c = counts[p]
        deficit = (target - current) / np.maximum(target, 1)  # déficit relatif par classe
        score = (deficit * c).sum(axis=1)
        k = int(np.argmax(score))
        assignment[p] = names[k]
        current[k] += c

    return [Sample(s.path, s.label, s.patient_id, assignment[s.patient_id]) for s in samples]


def write_manifest(samples: list[Sample], path: Path, root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "patient_id", "split"])
        for s in samples:
            p = Path(s.path)
            if root is not None:
                try:
                    p = p.relative_to(root)
                except ValueError:
                    pass
            w.writerow([p.as_posix(), s.label, s.patient_id, s.split])


def read_manifest(path: Path, root: Path | None = None) -> list[Sample]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        p = Path(r["path"])
        if root is not None and not p.is_absolute():
            p = root / p
        out.append(Sample(str(p), int(r["label"]), r["patient_id"], r["split"]))
    return out


def split_summary(samples: list[Sample]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for name in sorted({s.split for s in samples}):
        sub = [s for s in samples if s.split == name]
        summary[name] = {
            "images": len(sub),
            "patients": len({s.patient_id for s in sub}),
            "label_0": sum(s.label == 0 for s in sub),
            "label_1": sum(s.label == 1 for s in sub),
        }
    return summary
