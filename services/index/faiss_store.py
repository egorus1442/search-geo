"""
FAISS-индекс для поиска BoVW гистограмм.

Структура: IVF-Flat (инвертированные файлы с полными векторами).
Для MVP это оптимально: точный поиск в n_probe ячейках, умеренная память.

При росте базы > 500К патчей → переход на IVF-PQ (с квантизацией).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import get_logger, get_settings

logger = get_logger(__name__)
_s = get_settings()


class FaissStore:
    """
    Оборачивает FAISS IndexIVFFlat.

    Использование:
        store = FaissStore(dim=1024)
        store.train(all_histograms)
        store.add(histograms, ids)
        store.save()

        store = FaissStore.load()
        D, I = store.search(query_hist, k=100)
    """

    def __init__(
        self,
        dim: int | None = None,
        n_lists: int | None = None,
        n_probe: int | None = None,
        index_path: Path | None = None,
    ) -> None:
        self.dim = dim or _s.vocab_size
        self.n_lists = n_lists or _s.faiss_n_lists
        self.n_probe = n_probe or _s.faiss_n_probe
        self.index_path = Path(index_path or _s.faiss_index_path)
        self._index: faiss.Index | None = None  # type: ignore[name-defined]

    # ── Build ─────────────────────────────────────────────────────────────────

    def _create_index(self) -> faiss.IndexIDMap:  # type: ignore[name-defined]
        """
        IndexIDMap(IndexIVFFlat) — позволяет хранить произвольные int64 ID
        вместо автоинкрементных (нужно для маппинга на patch_id из PostgreSQL).
        """
        quantizer = faiss.IndexFlatL2(self.dim)
        ivf = faiss.IndexIVFFlat(quantizer, self.dim, self.n_lists, faiss.METRIC_L2)
        index = faiss.IndexIDMap(ivf)
        return index

    def _ivf(self) -> faiss.IndexIVFFlat:  # type: ignore[name-defined]
        """Достать внутренний IVF индекс для train/nprobe."""
        return faiss.downcast_index(self._index.index)  # type: ignore[union-attr]

    def train(self, vectors: np.ndarray) -> None:
        """
        Обучить IVF quantizer на всём наборе векторов.
        Требуется до add(). vectors: float32 (N, dim).
        """
        vectors = self._validate(vectors)
        if vectors.shape[0] < self.n_lists:
            self.n_lists = max(1, vectors.shape[0] // 4)
            logger.warning(
                "faiss_n_lists_reduced",
                new_n_lists=self.n_lists,
                reason="not enough training vectors",
            )

        self._index = self._create_index()
        logger.info("faiss_train_start", n_vectors=len(vectors), dim=self.dim, n_lists=self.n_lists)
        self._ivf().train(vectors)
        logger.info("faiss_train_done")

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Добавить векторы с внешними ID (patch_id из PostgreSQL).
        ids: int64 array (N,)
        """
        if self._index is None:
            raise RuntimeError("Call train() before add()")

        vectors = self._validate(vectors)
        ids = ids.astype(np.int64)
        self._index.add_with_ids(vectors, ids)
        logger.info("faiss_add", n_vectors=len(vectors), total=self._index.ntotal)

    def build_from_scratch(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Обучить + добавить за один вызов."""
        self.train(vectors)
        self.add(vectors, ids)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Поиск k ближайших соседей.

        query: float32 (dim,) или (1, dim)
        Returns: (distances, ids), each shape (k,)
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call train()+add() or load().")

        query = self._validate(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        self._ivf().nprobe = self.n_probe
        distances, ids = self._index.search(query, k)
        return distances[0], ids[0]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is None:
            raise RuntimeError("Nothing to save — index is empty.")
        faiss.write_index(self._index, str(path))
        logger.info("faiss_saved", path=str(path), ntotal=self._index.ntotal)
        return path

    @classmethod
    def load(cls, path: Path | None = None, n_probe: int | None = None) -> "FaissStore":
        path = Path(path or get_settings().faiss_index_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found at {path}")
        store = cls(index_path=path, n_probe=n_probe)
        store._index = faiss.read_index(str(path))
        store._index.nprobe = store.n_probe
        logger.info(
            "faiss_loaded",
            path=str(path),
            ntotal=store._index.ntotal,
            dim=store._index.d,
        )
        return store

    @property
    def ntotal(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def is_trained(self) -> bool:
        return self._index is not None and self._index.is_trained

    # ── Utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(vectors: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        if self._index is None:
            return {"status": "not_loaded"}
        return {
            "ntotal": self._index.ntotal,
            "dim": self._index.d,
            "n_lists": self.n_lists,
            "n_probe": self.n_probe,
            "is_trained": self._ivf().is_trained,
        }
