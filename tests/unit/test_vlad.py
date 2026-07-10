"""Тесты VLAD-энкодера и RootSIFT (offline, без внешнего стека)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from services.features.rootsift import to_rootsift
from services.features.vlad import VladEncoder


def _random_descriptors(n=120, dim=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, dim)) * 255).astype(np.float32)


def _stream_factory(n_patches=40):
    def factory():
        for i in range(n_patches):
            yield _random_descriptors(n=80, seed=i)
    return factory


class TestRootSIFT:
    def test_shape_preserved_and_l2_norm(self):
        desc = _random_descriptors(n=10)
        r = to_rootsift(desc)
        assert r.shape == desc.shape
        assert r.dtype == np.float32
        # RootSIFT: L1(desc) затем sqrt → L2-норма каждого вектора ≈ 1
        norms = np.linalg.norm(r, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_none_passthrough(self):
        assert to_rootsift(None) is None


class TestVladEncoder:
    def test_fit_and_encode_no_pca(self):
        enc = VladEncoder(n_centroids=16, use_pca=False)
        enc.fit(_stream_factory())
        assert enc.is_fitted
        assert enc.dim == 16 * 128

        vec = enc.encode(_random_descriptors(n=90, seed=99))
        assert vec.shape == (enc.dim,)
        assert vec.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_fit_and_encode_with_pca(self):
        enc = VladEncoder(n_centroids=16, use_pca=True, pca_dim=32)
        enc.fit(_stream_factory())
        assert enc.is_fitted
        assert enc.dim == 32

        vec = enc.encode(_random_descriptors(n=90, seed=7))
        assert vec.shape == (32,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_encode_none_returns_zeros_raw(self):
        enc = VladEncoder(n_centroids=8, use_pca=False)
        enc.fit(_stream_factory())
        vec = enc.encode(None)
        assert vec.shape == (enc.dim,)
        assert np.allclose(vec, 0.0)

    def test_same_input_same_output(self):
        enc = VladEncoder(n_centroids=16, use_pca=True, pca_dim=32)
        enc.fit(_stream_factory())
        desc = _random_descriptors(n=70, seed=3)
        np.testing.assert_array_almost_equal(enc.encode(desc), enc.encode(desc))

    def test_save_and_load(self):
        enc = VladEncoder(n_centroids=16, use_pca=True, pca_dim=24)
        enc.fit(_stream_factory())
        desc = _random_descriptors(n=60, seed=5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vlad.pkl"
            enc.save(path)
            loaded = VladEncoder.load(path)

        assert loaded.dim == enc.dim
        assert loaded.n_centroids == 16
        np.testing.assert_array_almost_equal(enc.encode(desc), loaded.encode(desc))

    def test_load_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            VladEncoder.load(Path("/nonexistent/vlad.pkl"))

    def test_encode_before_fit_raises(self):
        enc = VladEncoder(n_centroids=8)
        with pytest.raises(RuntimeError):
            enc.encode(_random_descriptors())
