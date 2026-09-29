"""E3 — Test de charge : SLO latence < 500 ms P95 sur /v1/predict.

Usage (API démarrée, conteneur limité à 2 vCPU / 2 Go) :
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 5 -t 2m --csv reports/load

Images : échantillons de data/processed (ou LOAD_IMAGES_DIR), sinon cellules synthétiques.
Clé API : variable LOAD_API_KEY. À 60 req/min par clé, prévoir une clé par utilisateur
simulé (LOAD_API_KEY=k1,k2,...) ou relever RATE_LIMIT côté API pendant le test.
Le test échoue (code de sortie 1) si le P95 dépasse 500 ms ou si des erreurs surviennent.
"""
from __future__ import annotations

import io
import itertools
import os
import random
from pathlib import Path

from locust import HttpUser, between, events, task

SLO_P95_MS = 500
ROOT = Path(__file__).resolve().parents[2]


def load_images() -> list[bytes]:
    folder = Path(os.environ.get("LOAD_IMAGES_DIR", ROOT / "data" / "raw" / "cell_images"))
    files = list(itertools.islice(folder.rglob("*.png"), 200)) if folder.exists() else []
    if files:
        return [f.read_bytes() for f in files]
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    out = []
    for _ in range(20):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (130, 130, 3), dtype=np.uint8)).save(buf, "PNG")
        out.append(buf.getvalue())
    return out


IMAGES = load_images()
KEYS = itertools.cycle([k for k in os.environ.get("LOAD_API_KEY", "").split(",") if k] or [None])


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        key = next(KEYS)
        self.headers = {"X-API-Key": key} if key else {}

    @task(10)
    def predict(self):
        self.client.post("/v1/predict", files={"file": ("cell.png", random.choice(IMAGES), "image/png")},
                         headers=self.headers, name="/v1/predict")

    @task(1)
    def health(self):
        self.client.get("/health")


@events.quitting.add_listener
def enforce_slo(environment, **_):
    stats = environment.stats.get("/v1/predict", "POST")
    p95 = stats.get_response_time_percentile(0.95)
    ok = p95 is not None and p95 <= SLO_P95_MS and stats.num_failures == 0
    print(f"[SLO] /v1/predict P95 = {p95} ms (cible <= {SLO_P95_MS}), échecs = {stats.num_failures} -> "
          f"{'OK' if ok else 'ÉCHEC'}")
    if not ok:
        environment.process_exit_code = 1
