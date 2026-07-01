"""Тесты BoVW словаря."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from services.features.vocabulary import Vocabulary


def _random_descriptors(n=100, dim=128) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype(np.float32)


def _descriptor_stream(n_patches=20):
    for _ in range(n_patches):
        yield _random_descriptors(n=50)


class TestVocabulary:
    def test_fit_and_encode(self):
        vocab = Vocabulary(vocab_size=64)
        vocab.fit(_descriptor_stream(n_patches=30))

        assert vocab.is_fitted

        desc = _random_descriptors(n=80)
        hist = vocab.encode(desc)

        assert hist.shape == (64,)
        assert hist.dtype == np.float32
        # L2-нормированный вектор: норма ≈ 1 (если ненулевой)
        norm = np.linalg.norm(hist)
        assert norm == pytest.approx(1.0, abs=1e-5) or norm == pytest.approx(0.0, abs=1e-5)

    def test_encode_none_returns_zeros(self):
        vocab = Vocabulary(vocab_size=64)
        vocab.fit(_descriptor_stream())
        hist = vocab.encode(None)
        assert np.all(hist == 0)
        assert hist.shape == (64,)

    def test_save_and_load(self):
        vocab = Vocabulary(vocab_size=32)
        vocab.fit(_descriptor_stream())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.pkl"
            vocab.save(path)

            loaded = Vocabulary.load(path)
            assert loaded.vocab_size == 32
            assert loaded.is_fitted

            desc = _random_descriptors(n=50)
            h1 = vocab.encode(desc)
            h2 = loaded.encode(desc)
            np.testing.assert_array_almost_equal(h1, h2)

    def test_load_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            Vocabulary.load(Path("/nonexistent/vocab.pkl"))

    def test_encode_before_fit_raises(self):
        vocab = Vocabulary(vocab_size=32)
        with pytest.raises(RuntimeError):
            vocab.encode(_random_descriptors())
