"""A1 — Récupération du dataset NIH (~27 558 images, 2 classes équilibrées).

Usage :
    python -m src.data.download                          # télécharge depuis le NIH
    python -m src.data.download --source archive.zip     # utilise une archive locale
    python -m src.data.download --source dossier/        # utilise un dossier déjà extrait

Le résultat est normalisé dans `data.raw_dir` :  <raw_dir>/Parasitized, <raw_dir>/Uninfected.
Les archives Kaggle contiennent un doublon imbriqué (cell_images/cell_images) : il est ignoré.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.config import load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)
CLASSES = ("Parasitized", "Uninfected")


def find_class_root(base: Path) -> Path:
    """Premier dossier (le moins profond) contenant Parasitized/ et Uninfected/."""
    candidates = [base] + sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: len(p.parts))
    for c in candidates:
        if all((c / cls).is_dir() for cls in CLASSES):
            return c
    raise FileNotFoundError(f"Aucun dossier contenant {CLASSES} sous {base}")


def copy_dataset(src_root: Path, dest: Path) -> dict[str, int]:
    counts = {}
    for cls in CLASSES:
        (dest / cls).mkdir(parents=True, exist_ok=True)
        files = [p for p in (src_root / cls).iterdir() if p.suffix.lower() == ".png"]
        for p in files:
            target = dest / cls / p.name
            if not target.exists():
                shutil.copy2(p, target)
        counts[cls] = len(files)
    return counts


def download(url: str, target: Path) -> None:
    log.info("Téléchargement de %s ...", url)
    with urllib.request.urlopen(url) as resp, open(target, "wb") as out:  # noqa: S310 (URL de config)
        shutil.copyfileobj(resp, out)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="Archive .zip ou dossier local (sinon téléchargement NIH)")
    parser.add_argument("--dest", default=cfg["data"]["raw_dir"])
    args = parser.parse_args()

    dest = resolve_path(args.dest)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = Path(args.source) if args.source else None
        if source is None:
            source = tmp / "cell_images.zip"
            download(cfg["data"]["nih_url"], source)
        if source.is_file():
            log.info("Extraction de %s ...", source)
            with zipfile.ZipFile(source) as zf:
                zf.extractall(tmp / "extracted")
            source = tmp / "extracted"
        counts = copy_dataset(find_class_root(source), dest)
    log.info("Dataset prêt dans %s : %s", dest, counts)


if __name__ == "__main__":
    main()
