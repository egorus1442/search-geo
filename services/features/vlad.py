"""
VLAD (Vector of Locally Aggregated Descriptors) поверх RootSIFT.

Зачем вместо BoVW
-----------------
BoVW кодирует патч как tf-idf гистограмму «сколько раз встретилось каждое
визуальное слово» — жёсткое присвоение + сумма счётчиков. Это самый грубый
способ агрегации: он теряет всё, кроме частот, и потому (а) плохо различает
разные участки одного региона с похожим ландшафтом и (б) сильно страдает от
domain gap UAV↔Sentinel (смещение «слов» → другая гистограмма у того же места).

VLAD агрегирует не счётчики, а *остатки* дескрипторов относительно ближайшего
центроида: для каждого центроида c_i суммируем (x - c_i) по всем дескрипторам,
назначенным на c_i. Получается вектор n_centroids * 128, который сохраняет
информацию о том, КАК именно локальные признаки отклоняются от типовых —
гораздо различимее гистограммы при том же (или меньшем) числе центроидов.

Пост-обработка (важна для качества):
  1. intra-normalization — L2 по каждому блоку центроида (борется с burstiness:
     доминирующая текстура не забивает вектор).
  2. power-law (signed sqrt) + глобальная L2 — стандартный VLAD-«whitening lite».
  3. PCA-whitening (опц.) — декоррелирует и сжимает вектор; на retrieval это
     ещё один заметный прирост, плюс компактнее для FAISS.

RootSIFT применяется к дескрипторам ВНУТРИ энкодера (см. rootsift.py) — и при
обучении центроидов, и при encode, чтобы query и база жили в одном пространстве.

Интерфейс совместим с Vocabulary (encode/save/load/dim/is_fitted), поэтому
энкодеры взаимозаменяемы за фабрикой services.features.coarse.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from config import get_logger, get_settings
from services.features.rootsift import to_rootsift

logger = get_logger(__name__)
_s = get_settings()

SIFT_DIM = 128

# Фабрика заново-итерируемого потока дескрипторов: каждый вызов возвращает
# свежий генератор массивов (N_i, 128) сырых SIFT-дескрипторов, по одному на
# патч. Нужна именно фабрика (а не один генератор), т.к. fit() проходит по
# базе дважды: сначала k-means, затем PCA.
DescStreamFactory = Callable[[], Iterator[np.ndarray]]


class VladEncoder:
    def __init__(
        self,
        n_centroids: int | None = None,
        use_pca: bool | None = None,
        pca_dim: int | None = None,
    ) -> None:
        self.n_centroids = n_centroids or _s.vlad_n_centroids
        self.use_pca = _s.vlad_use_pca if use_pca is None else use_pca
        self.pca_dim = pca_dim or _s.vlad_pca_dim
        self._kmeans: MiniBatchKMeans | None = None
        self._pca: PCA | None = None

    # ── Dimensions ──────────────────────────────────────────────────────────────

    @property
    def raw_dim(self) -> int:
        return self.n_centroids * SIFT_DIM

    @property
    def dim(self) -> int:
        """Итоговая размерность вектора, который уходит в FAISS."""
        if self.use_pca and self._pca is not None:
            return int(self._pca.n_components_)
        return self.raw_dim

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None and (not self.use_pca or self._pca is not None)

    # ── Training ────────────────────────────────────────────────────────────────

    def fit(
        self,
        stream_factory: DescStreamFactory,
        n_sample_per_patch: int | None = None,
    ) -> "VladEncoder":
        """
        Обучить энкодер за два прохода по базе:
          1) k-means центроидов на подвыборке RootSIFT-дескрипторов;
          2) (если use_pca) PCA-whitening на VLAD-векторах всех патчей.
        """
        self._fit_centroids(stream_factory, n_sample_per_patch)
        if self.use_pca:
            self._fit_pca(stream_factory)
        logger.info("vlad_fit_done", n_centroids=self.n_centroids, dim=self.dim)
        return self

    def _fit_centroids(
        self,
        stream_factory: DescStreamFactory,
        n_sample_per_patch: int | None,
    ) -> None:
        n_sample = n_sample_per_patch or _s.vlad_sample_per_patch
        sampled: list[np.ndarray] = []
        n_patches = 0
        for descs in stream_factory():
            if descs is None or len(descs) == 0:
                continue
            r = to_rootsift(descs)
            idx = np.random.choice(len(r), size=min(n_sample, len(r)), replace=False)
            sampled.append(r[idx])
            n_patches += 1

        if not sampled:
            raise ValueError("No descriptors provided for VLAD training")

        all_descs = np.vstack(sampled).astype(np.float32)
        logger.info(
            "vlad_kmeans_start",
            n_descriptors=len(all_descs),
            n_centroids=self.n_centroids,
            n_patches=n_patches,
        )
        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_centroids,
            n_init=_s.vlad_kmeans_n_init,
            batch_size=min(10_000, len(all_descs)),
            random_state=42,
            verbose=0,
        )
        self._kmeans.fit(all_descs)

    def _fit_pca(self, stream_factory: DescStreamFactory) -> None:
        vlad_vectors: list[np.ndarray] = []
        for descs in stream_factory():
            v = self._encode_raw(descs)
            if v is not None:
                vlad_vectors.append(v)

        if not vlad_vectors:
            raise ValueError("No VLAD vectors computed for PCA fitting")

        matrix = np.vstack(vlad_vectors).astype(np.float32)
        n_components = min(self.pca_dim, matrix.shape[0], matrix.shape[1])
        logger.info("vlad_pca_start", n_samples=matrix.shape[0], n_components=n_components)
        self._pca = PCA(n_components=n_components, whiten=True, random_state=42)
        self._pca.fit(matrix)

    # ── Inference ───────────────────────────────────────────────────────────────

    def _encode_raw(self, descriptors: np.ndarray | None) -> np.ndarray:
        """VLAD-вектор ДО PCA (raw_dim,), L2-нормированный. Нулевой для пустого входа."""
        if self._kmeans is None:
            raise RuntimeError("VladEncoder not fitted. Call fit() or load() first.")

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.raw_dim, dtype=np.float32)

        r = to_rootsift(descriptors).astype(np.float32)
        assignments = self._kmeans.predict(r)
        centroids = self._kmeans.cluster_centers_

        vlad = np.zeros((self.n_centroids, SIFT_DIM), dtype=np.float32)
        for i in range(self.n_centroids):
            mask = assignments == i
            if np.any(mask):
                vlad[i] = (r[mask] - centroids[i]).sum(axis=0)

        # intra-normalization (per-centroid L2) — душит burstiness
        block_norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        vlad = vlad / (block_norms + 1e-12)

        v = vlad.flatten()
        # power-law + глобальная L2
        v = np.sign(v) * np.sqrt(np.abs(v))
        v /= (np.linalg.norm(v) + 1e-12)
        return v.astype(np.float32)

    def encode(self, descriptors: np.ndarray | None) -> np.ndarray:
        """
        Дескрипторы патча → финальный coarse-вектор (dim,), L2-нормированный.
        Совместимо по сигнатуре с Vocabulary.encode.
        """
        v = self._encode_raw(descriptors)
        if self.use_pca and self._pca is not None:
            v = self._pca.transform(v.reshape(1, -1))[0].astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
        return v.astype(np.float32)

    # ── Serialization ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or _s.vlad_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "kmeans": self._kmeans,
                    "pca": self._pca,
                    "n_centroids": self.n_centroids,
                    "use_pca": self.use_pca,
                    "pca_dim": self.pca_dim,
                },
                f,
            )
        logger.info("vlad_saved", path=str(path), dim=self.dim)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "VladEncoder":
        path = Path(path or _s.vlad_path)
        if not path.exists():
            raise FileNotFoundError(f"VLAD encoder not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        enc = cls(
            n_centroids=data["n_centroids"],
            use_pca=data["use_pca"],
            pca_dim=data["pca_dim"],
        )
        enc._kmeans = data["kmeans"]
        enc._pca = data["pca"]
        logger.info("vlad_loaded", path=str(path), dim=enc.dim)
        return enc
