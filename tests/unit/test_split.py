import numpy as np
import pytest

from src.data.split import (
    Sample,
    group_stratified_split,
    list_samples,
    patient_id_from_filename,
    read_manifest,
    split_summary,
    write_manifest,
)

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def synthetic_samples(n_patients=200, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for p in range(n_patients):
        n_pos = int(rng.integers(0, 150)) if p % 4 else 0
        n_neg = int(rng.integers(20, 150))
        for label, n in ((1, n_pos), (0, n_neg)):
            samples += [Sample(f"P{p}_IMG_{label}_{i}.png", label, f"P{p}") for i in range(n)]
    return samples


def test_patient_id_from_nih_filename():
    assert patient_id_from_filename("C100P61ThinF_IMG_20150918_144104_cell_162.png") == "C100P61ThinF"
    assert patient_id_from_filename("dir/C1_thinF_IMG_2015_cell_9.png") == "C1_thinF"
    assert patient_id_from_filename("lab42_cell_7.png") == "lab42_cell"


def test_no_patient_leak_between_splits():
    out = group_stratified_split(synthetic_samples(), FRACTIONS, seed=42)
    by_patient = {}
    for s in out:
        by_patient.setdefault(s.patient_id, set()).add(s.split)
    assert all(len(v) == 1 for v in by_patient.values())


def test_split_proportions_and_class_balance():
    out = group_stratified_split(synthetic_samples(), FRACTIONS, seed=42)
    summary = split_summary(out)
    total = len(out)
    for name, frac in FRACTIONS.items():
        assert summary[name]["images"] / total == pytest.approx(frac, abs=0.02)
        pos_share = summary[name]["label_1"] / summary[name]["images"]
        overall = sum(s.label for s in out) / total
        assert pos_share == pytest.approx(overall, abs=0.05)


def test_split_is_reproducible_and_seed_dependent():
    samples = synthetic_samples()
    a = group_stratified_split(samples, FRACTIONS, seed=1)
    b = group_stratified_split(samples, FRACTIONS, seed=1)
    c = group_stratified_split(samples, FRACTIONS, seed=2)
    assert [s.split for s in a] == [s.split for s in b]
    assert [s.split for s in a] != [s.split for s in c]


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        group_stratified_split(synthetic_samples(10), {"train": 0.5, "val": 0.2}, seed=0)


def test_list_samples_and_manifest_roundtrip(tmp_path, cfg):
    from tests.conftest import make_cell

    for cls in cfg["class_map"].values():
        (tmp_path / cls).mkdir()
        make_cell(cls == "Parasitized").save(tmp_path / cls / f"P1_IMG_{cls}.png")
    (tmp_path / "Uninfected" / "Thumbs.db").write_bytes(b"x")
    samples = list_samples(tmp_path, cfg["class_map"])
    assert len(samples) == 2
    assert {s.label for s in samples} == {0, 1}
    write_manifest(samples, tmp_path / "m.csv", root=tmp_path)
    back = read_manifest(tmp_path / "m.csv", root=tmp_path)
    assert [(s.path, s.label) for s in back] == [(s.path, s.label) for s in samples]
