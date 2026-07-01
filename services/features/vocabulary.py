"""
BoVW Visual Vocabulary.

Обучение: K-Means на подборке SIFT-дескрипторов из базы.
Сохранение/загрузка через pickle.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import get_logger, get_settings

logger = get_logger(__name__)
_s = get_settings()


class Vocabulary:
    """
    Visual Vocabulary (кластерные центроиды K-Means) + IDF веса.

    Методы:
        fit(descriptors)          — обучить на массиве дескрипторов
        predict(descriptors)      — назначить слова дескрипторам
        encode(descriptors)       — патч → tf-idf BoVW гистограмма
        save(path) / load(path)   — сериализация
    """

    def __init__(self, vocab_size: int | None = None) -> None:
        self.vocab_size = vocab_size or _s.vocab_size
        self._kmeans: MiniBatchKMeans | None = None
        self._idf: np.ndarray | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        descriptor_stream: Iterator[np.ndarray],
        n_sample_per_patch: int | None = None,
    ) -> "Vocabulary":
        """
        Обучить словарь.

        descriptor_stream: итератор массивов дескрипторов (N_i, 128), по одному на патч.
        n_sample_per_patch: сколько дескрипторов сэмплировать из каждого патча.
        """
        n_sample = n_sample_per_patch or _s.vocab_sample_per_patch
        sampled: list[np.ndarray] = []

        total_patches = 0
        for descs in descriptor_stream:
            if descs is None or len(descs) == 0:
                continue
            idx = np.random.choice(len(descs), size=min(n_sample, len(descs)), replace=False)
            sampled.append(descs[idx])
            total_patches += 1

        if not sampled:
            raise ValueError("No descriptors provided for vocabulary training")

        all_descs = np.vstack(sampled).astype(np.float32)
        logger.info(
            "vocab_fit_start",
            n_descriptors=len(all_descs),
            vocab_size=self.vocab_size,
            n_patches=total_patches,
        )

        self._kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            n_init=_s.vocab_kmeans_n_init,
            batch_size=min(10_000, len(all_descs)),
            random_state=42,
            verbose=0,
        )
        self._kmeans.fit(all_descs)

        # Вычислить IDF: для каждого слова — в скольких патчах встречается
        logger.info("vocab_computing_idf", n_patches=total_patches)
        word_doc_count = np.zeros(self.vocab_size, dtype=np.float64)

        # Повторный проход по сохранённым сэмплам для IDF
        # (упрощение: считаем по sampled батчам, не по полной базе)
        for patch_descs in sampled:
            words = self._kmeans.predict(patch_descs.astype(np.float32))
            unique_words = np.unique(words)
            word_doc_count[unique_words] += 1

        n_docs = float(total_patches)
        self._idf = np.log(n_docs / (word_doc_count + 1.0)) + 1.0  # smooth IDF

        logger.info("vocab_fit_done", vocab_size=self.vocab_size)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        """Назначить визуальные слова дескрипторам. Returns word_ids array."""
        if self._kmeans is None:
            raise RuntimeError("Vocabulary not trained. Call fit() or load() first.")
        return self._kmeans.predict(descriptors.astype(np.float32))

    def encode(self, descriptors: np.ndarray | None) -> np.ndarray:
        """
        Дескрипторы патча → tf-idf взвешенная BoVW гистограмма.

        Returns: float32 array shape (vocab_size,), L2-нормированная.
        """
        if self._kmeans is None or self._idf is None:
            raise RuntimeError("Vocabulary not fitted.")

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size, dtype=np.float32)

        word_ids = self.predict(descriptors)
        tf = np.bincount(word_ids, minlength=self.vocab_size).astype(np.float64)
        tf /= max(len(word_ids), 1)  # term frequency

        hist = (tf * self._idf).astype(np.float32)

        norm = np.linalg.norm(hist)
        if norm > 1e-8:
            hist /= norm
        return hist

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = path or _s.vocab_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"kmeans": self._kmeans, "idf": self._idf, "vocab_size": self.vocab_size}, f)
        logger.info("vocab_saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Vocabulary":
        path = path or _s.vocab_path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls(vocab_size=data["vocab_size"])
        vocab._kmeans = data["kmeans"]
        vocab._idf = data["idf"]
        logger.info("vocab_loaded", path=str(path), vocab_size=vocab.vocab_size)
        return vocab

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None and self._idf is not None
