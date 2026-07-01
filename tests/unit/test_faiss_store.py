"""Тесты FAISS индекса."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from services.index.faiss_store import FaissStore


def _random_vecs(n=200, dim=64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, dim)).astype(np.float32)


class TestFaissStore:
    DIM = 64

    def test_build_and_search(self):
        vecs = _random_vecs(200, self.DIM)
        ids = np.arange(200, dtype=np.int64)

        store = FaissStore(dim=self.DIM, n_lists=8, n_probe=4)
        store.build_from_scratch(vecs, ids)

        assert store.is_trained
        assert store.ntotal == 200

        query = vecs[0]
        distances, result_ids = store.search(query, k=5)

        assert len(distances) == 5
        assert len(result_ids) == 5
        # Ближайший сосед к самому себе — должен быть он же
        assert result_ids[0] == 0

    def test_search_returns_valid_ids(self):
        vecs = _random_vecs(100, self.DIM)
        ids = np.arange(1000, 1100, dtype=np.int64)  # произвольные ID

        store = FaissStore(dim=self.DIM, n_lists=4, n_probe=2)
        store.build_from_scratch(vecs, ids)

        _, result_ids = store.search(vecs[5], k=10)
        valid = result_ids[result_ids != -1]
        for rid in valid:
            assert 1000 <= rid < 1100

    def test_save_and_load(self):
        vecs = _random_vecs(150, self.DIM)
        ids = np.arange(150, dtype=np.int64)

        store = FaissStore(dim=self.DIM, n_lists=4, n_probe=2)
        store.build_from_scratch(vecs, ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.faiss"
            store.save(path)

            loaded = FaissStore.load(path, n_probe=2)
            assert loaded.ntotal == 150

            _, orig_ids = store.search(vecs[0], k=3)
            _, load_ids = loaded.search(vecs[0], k=3)
            np.testing.assert_array_equal(orig_ids, load_ids)

    def test_search_before_train_raises(self):
        store = FaissStore(dim=self.DIM)
        with pytest.raises(RuntimeError):
            store.search(np.zeros(self.DIM, dtype=np.float32))

    def test_add_before_train_raises(self):
        store = FaissStore(dim=self.DIM)
        with pytest.raises(RuntimeError):
            store.add(np.zeros((5, self.DIM), dtype=np.float32), np.arange(5, dtype=np.int64))

    def test_n_lists_auto_reduced(self):
        """Если данных меньше n_lists — n_lists уменьшается автоматически."""
        vecs = _random_vecs(10, self.DIM)
        ids = np.arange(10, dtype=np.int64)
        store = FaissStore(dim=self.DIM, n_lists=256, n_probe=4)
        store.build_from_scratch(vecs, ids)
        assert store.ntotal == 10
