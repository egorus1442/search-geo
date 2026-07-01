"""Тесты SIFT экстрактора."""
import numpy as np
import pytest
from PIL import Image
import io

from services.features.sift import extract_descriptors, load_image_gray


def _make_png(width=256, height=256, noise=True) -> bytes:
    """Сгенерировать тестовое изображение с текстурой."""
    rng = np.random.default_rng(42)
    if noise:
        arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    else:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_extract_descriptors_returns_array():
    img_bytes = _make_png(noise=True)
    kp, desc = extract_descriptors(img_bytes)
    assert desc is not None
    assert desc.ndim == 2
    assert desc.shape[1] == 128          # SIFT дескриптор = 128 dim
    assert desc.dtype == np.float32
    assert len(kp) == desc.shape[0]


def test_extract_descriptors_blank_image_returns_none():
    """Однородное изображение — нет ключевых точек."""
    img_bytes = _make_png(noise=False)
    kp, desc = extract_descriptors(img_bytes)
    assert desc is None
    assert kp == []


def test_load_image_gray_from_bytes():
    img_bytes = _make_png()
    gray = load_image_gray(img_bytes)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_load_image_gray_from_ndarray():
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gray = load_image_gray(arr)
    assert gray.ndim == 2


def test_extract_descriptors_resize():
    """Большое изображение должно быть ресайзнуто."""
    img_bytes = _make_png(width=2048, height=2048)
    kp, desc = extract_descriptors(img_bytes, max_side=512)
    assert desc is not None
