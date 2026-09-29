from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import create_app
from src.api.service import ModelService
from src.api.settings import Settings
from src.config import load_config
from src.models.inference import MockPredictor


def make_cell(infected: bool, size: int = 140, seed: int = 0) -> Image.Image:
    """Cellule synthétique : disque rose sur fond noir, + taches sombres si 'infectée'."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    c = size / 2
    disk = (yy - c) ** 2 + (xx - c) ** 2 < (size * 0.42) ** 2
    img = np.zeros((size, size, 3), np.float32)
    img[disk] = [0.85, 0.62, 0.66]
    img += rng.normal(0, 0.02, img.shape) * disk[..., None]
    if infected:
        for _ in range(3):
            y, x = rng.integers(int(c - 25), int(c + 25), 2)
            spot = (yy - y) ** 2 + (xx - x) ** 2 < 49
            img[spot] = [0.35, 0.15, 0.45]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def infected_png() -> bytes:
    return to_bytes(make_cell(True))


@pytest.fixture
def healthy_png() -> bytes:
    return to_bytes(make_cell(False))


def mock_service(cfg, threshold: float = 0.5) -> ModelService:
    return ModelService(MockPredictor(), "1.0.0", threshold, cfg["class_map"], cfg["positive_class"],
                        cfg["preprocessing"]["image_size"], {"min_side": 32, "min_std": 0.01})


def make_client(cfg, **overrides) -> TestClient:
    settings = Settings(model_path="unused.onnx", serve_web_app=False, **overrides)
    return TestClient(create_app(settings, service=mock_service(cfg)))


@pytest.fixture
def client(cfg):
    with make_client(cfg, rate_limit="1000/minute") as c:
        yield c
