import io

import numpy as np
import pytest
from PIL import Image

from src.data.preprocessing import (
    InvalidImageError,
    check_quality,
    decode_image,
    load_image_file,
    preprocess,
    preprocess_uint8,
    to_rgb,
    uint8_to_model_input,
)
from tests.conftest import make_cell, to_bytes


def test_preprocess_shape_dtype_range():
    x = preprocess(make_cell(True), 128)
    assert x.shape == (128, 128, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_training_cache_matches_inference_preprocessing():
    """Non-régression : le cache d'entraînement (uint8) == prétraitement de l'API."""
    img = make_cell(True, size=97)
    np.testing.assert_allclose(uint8_to_model_input(preprocess_uint8(img)), preprocess(img), atol=1e-7)


def test_preprocessing_reference_values_are_stable():
    """Non-régression : toute modification du prétraitement fait échouer ce test."""
    img = Image.fromarray(np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) * 5)
    x = preprocess(img, 2)
    assert x.shape == (2, 2, 3)
    np.testing.assert_allclose(x[0, 0], [0.2118, 0.2314, 0.2510], atol=1e-3)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_decode_accepts_png_and_jpeg(fmt):
    img = decode_image(to_bytes(make_cell(False), fmt))
    assert img.mode == "RGB"


def test_decode_rejects_other_formats():
    with pytest.raises(InvalidImageError, match="non supporté"):
        decode_image(to_bytes(make_cell(False), "GIF"))


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50])
def test_decode_rejects_corrupt_data(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_decode_rejects_truncated_png():
    data = to_bytes(make_cell(True))
    with pytest.raises(InvalidImageError):
        decode_image(data[: len(data) // 2])


def test_rgba_composited_on_black_background():
    rgba = Image.new("RGBA", (4, 4), (255, 255, 255, 0))
    assert np.asarray(to_rgb(rgba)).max() == 0


def test_grayscale_converted_to_rgb():
    gray = Image.new("L", (40, 40), 128)
    assert preprocess(decode_image(to_bytes(gray)), 128).shape == (128, 128, 3)


def test_quality_flags_small_and_blank_images():
    assert check_quality(make_cell(True)).ok
    report = check_quality(Image.new("RGB", (10, 10)))
    assert not report.ok and len(report.issues) == 2


def test_load_image_file(tmp_path):
    p = tmp_path / "c.png"
    make_cell(True).save(p)
    assert load_image_file(p).size == (140, 140)


def test_decode_does_not_trust_extension():
    buf = io.BytesIO()
    make_cell(True).save(buf, format="BMP")
    with pytest.raises(InvalidImageError):
        decode_image(buf.getvalue())
