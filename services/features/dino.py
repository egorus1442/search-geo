"""
DINOv2 глобальные дескрипторы как coarse-метод retrieval'а.

Две головы за одним image-based контрактом:
  - DinoEncoder      — pooled DINOv2 эмбеддинг (cls | mean | gem),
                       dim = embed_dim бэкбона (у ViT-S/14 = 384).
  - DinoVladEncoder  — AnyLoc: VLAD-агрегация DINOv2 patch-токенов (token-facet),
                       dim = n_centroids * embed_dim.

В отличие от VLAD/BoVW (которые кодируют SIFT-дескрипторы), эти энкодеры
работают с САМИМ изображением: `input_kind = "image"`, методы
`encode_image` / `encode_image_batch`. Фабрика services.features.coarse и
consumers (index_task/localize/eval_recall) диспетчеризуют по `input_kind`.

torch/timm импортируются ЛЕНИВО внутри бэкбона, чтобы обычная сборка сервиса
(COARSE_METHOD=vlad|bovw) не тянула тяжёлые зависимости.

Почему это может помочь именно нашей задаче (UAV↔Sentinel):
  - DINOv2 кодирует «место» семантически, а не набор SIFT-текстур → верный
    патч сохраняет отрыв от фона на больших площадях (где валится VLAD/BoVW);
  - устойчивее к domain gap (сенсор/сезон/масштаб) и к низкотекстурному
    контенту (лес/поля), где SIFT не находит опорных точек.

Замечание про AnyLoc: оригинал использует value-facet конкретного слоя. Здесь
берём token-facet последнего блока (patch-токены из forward_features) — сильно
проще (без хуков на attention) и практически достаточно; если понадобится,
value-facet можно добавить отдельно.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Sequence, Union

import cv2
import numpy as np

from config import get_logger, get_settings
from services.features.sift import load_image

logger = get_logger(__name__)
_s = get_settings()

ImageSource = Union[bytes, np.ndarray, Path, str]
# Заново-итерируемая фабрика потока изображений (аналог DescStreamFactory в
# vlad.py): каждый вызов возвращает свежий генератор источников изображений.
ImageStreamFactory = Callable[[], Iterator[ImageSource]]


# ── helpers ────────────────────────────────────────────────────────────────

def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / (n + 1e-12)).astype(np.float32)


def _l2(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


# ── DINOv2 backbone (общий, кэшируется по model_name+image_size) ────────────

class _DinoBackbone:
    """
    Тонкая обёртка над timm-моделью DINOv2. Держит модель в eval на CPU,
    нормализацию входа и извлечение токенов. Один экземпляр переиспользуется
    обеими головами (pooling и VLAD) через _get_backbone (lru_cache) — чтобы
    не грузить веса дважды.
    """

    def __init__(self, model_name: str, image_size: int) -> None:
        import timm  # ленивый импорт тяжёлых зависимостей
        import torch

        self._torch = torch
        n_threads = int(_s.torch_num_threads or 0)
        if n_threads > 0:
            torch.set_num_threads(n_threads)

        self.model_name = model_name
        self.image_size = int(image_size)

        # img_size переинтерполирует позиционные эмбеддинги под наш размер
        # (у dinov2 pretrained img_size крупнее — 224 дешевле для CPU).
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=self.image_size
        )
        self.model.eval()

        # mean/std из pretrained-конфига модели (DINOv2 = ImageNet-нормализация);
        # на любой несовместимости версий timm откатываемся к ImageNet-дефолтам.
        try:
            cfg = timm.data.resolve_data_config({}, model=self.model)
            mean = cfg.get("mean", (0.485, 0.456, 0.406))
            std = cfg.get("std", (0.229, 0.224, 0.225))
        except Exception:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.embed_dim = int(getattr(self.model, "embed_dim", _s.global_descriptor_dim))
        self.num_prefix = int(getattr(self.model, "num_prefix_tokens", 1))

        logger.info(
            "dino_backbone_loaded",
            model=model_name,
            image_size=self.image_size,
            embed_dim=self.embed_dim,
            num_prefix=self.num_prefix,
        )

    def _to_chw(self, source: ImageSource) -> np.ndarray:
        img = load_image(source)  # BGR uint8 (общий загрузчик проекта)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x.transpose(2, 0, 1)  # (3, H, W)

    def _forward_tokens(self, sources: Sequence[ImageSource]):
        """(B, T, C) токены последнего блока (включая prefix-токены)."""
        torch = self._torch
        batch = np.stack([self._to_chw(s) for s in sources], axis=0)
        x = torch.from_numpy(batch)
        with torch.no_grad():
            feats = self.model.forward_features(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

    def pooled(self, sources: Sequence[ImageSource], pooling: str, gem_p: float) -> np.ndarray:
        """(N, embed_dim) L2-нормированные pooled-эмбеддинги."""
        feats = self._forward_tokens(sources)
        cls = feats[:, 0]
        patches = feats[:, self.num_prefix:]
        pooling = pooling.lower()
        if pooling == "cls":
            vec = cls
        elif pooling == "mean":
            vec = patches.mean(dim=1)
        else:  # gem
            p = float(gem_p)
            vec = patches.clamp(min=1e-6).pow(p).mean(dim=1).pow(1.0 / p)
        return _l2_normalize_rows(vec.cpu().numpy().astype(np.float32))

    def patch_tokens(self, source: ImageSource) -> np.ndarray:
        """(T, C) patch-токены одного изображения (для AnyLoc-VLAD)."""
        feats = self._forward_tokens([source])
        return feats[0, self.num_prefix:].cpu().numpy().astype(np.float32)


@lru_cache(maxsize=2)
def _get_backbone(model_name: str, image_size: int) -> _DinoBackbone:
    return _DinoBackbone(model_name, image_size)


# ── Head 1: pooled DINOv2 (COARSE_METHOD=dino) ──────────────────────────────

class DinoEncoder:
    """Pooled DINOv2 глобальный дескриптор. Обучение не требуется (pretrained)."""

    input_kind = "image"

    def __init__(
        self,
        model_name: str | None = None,
        image_size: int | None = None,
        pooling: str | None = None,
        gem_p: float | None = None,
    ) -> None:
        self.model_name = model_name or _s.global_model_name
        self.image_size = int(image_size or _s.global_image_size)
        self.pooling = (pooling or _s.global_pooling).lower()
        self.gem_p = float(gem_p if gem_p is not None else _s.global_gem_p)
        self._backbone: _DinoBackbone | None = None

    @property
    def _bb(self) -> _DinoBackbone:
        if self._backbone is None:
            self._backbone = _get_backbone(self.model_name, self.image_size)
        return self._backbone

    @property
    def dim(self) -> int:
        if self._backbone is not None:
            return self._backbone.embed_dim
        return int(_s.global_descriptor_dim)

    @property
    def is_fitted(self) -> bool:
        return True  # pretrained, отдельного обучения нет

    # ── training (no-op, для единообразия с фабрикой) ──────────────────────
    def fit_images(self, stream_factory: ImageStreamFactory, **_: object) -> "DinoEncoder":
        return self

    # ── inference ──────────────────────────────────────────────────────────
    def encode_image(self, source: ImageSource) -> np.ndarray:
        return self._bb.pooled([source], self.pooling, self.gem_p)[0]

    def encode_image_batch(self, sources: Sequence[ImageSource]) -> np.ndarray:
        if not sources:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._bb.pooled(sources, self.pooling, self.gem_p)

    def encode(self, descriptors: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError(
            "DinoEncoder работает с изображением: используй encode_image/encode_image_batch"
        )

    # ── serialization (сохраняем конфиг; веса берутся из timm-кэша) ────────
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or _s.global_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "type": "dino",
                    "model_name": self.model_name,
                    "image_size": self.image_size,
                    "pooling": self.pooling,
                    "gem_p": self.gem_p,
                },
                f,
            )
        logger.info("dino_saved", path=str(path), pooling=self.pooling)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DinoEncoder":
        path = Path(path or _s.global_model_path)
        if not path.exists():
            raise FileNotFoundError(f"DINO encoder not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("type") != "dino":
            raise ValueError(f"{path} is not a DinoEncoder model (type={data.get('type')!r})")
        enc = cls(
            model_name=data["model_name"],
            image_size=data["image_size"],
            pooling=data["pooling"],
            gem_p=data["gem_p"],
        )
        logger.info("dino_loaded", path=str(path), pooling=enc.pooling)
        return enc


# ── Head 2: AnyLoc (DINOv2 patch-токены + VLAD) — COARSE_METHOD=dino_vlad ────

class DinoVladEncoder:
    """
    VLAD поверх DINOv2 patch-токенов (упрощённый AnyLoc, token-facet).

    Обучение (fit_images): один проход по патчам — сэмплируем patch-токены,
    учим k-means центроидов. encode_image агрегирует остатки токенов к
    ближайшим центроидам (та же математика, что в services/features/vlad.py,
    но без RootSIFT — он специфичен для SIFT).
    """

    input_kind = "image"

    def __init__(
        self,
        model_name: str | None = None,
        image_size: int | None = None,
        n_centroids: int | None = None,
        sample_per_patch: int | None = None,
    ) -> None:
        self.model_name = model_name or _s.global_model_name
        self.image_size = int(image_size or _s.global_image_size)
        self.n_centroids = int(n_centroids or _s.global_vlad_n_centroids)
        self.sample_per_patch = int(sample_per_patch or _s.global_vlad_sample_per_patch)
        self._kmeans = None
        self._backbone: _DinoBackbone | None = None

    @property
    def _bb(self) -> _DinoBackbone:
        if self._backbone is None:
            self._backbone = _get_backbone(self.model_name, self.image_size)
        return self._backbone

    @property
    def dim(self) -> int:
        return self.n_centroids * self._bb.embed_dim

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    # ── training ─────────────────────────────────────────────────────────
    def fit_images(
        self,
        stream_factory: ImageStreamFactory,
        n_sample_per_patch: int | None = None,
    ) -> "DinoVladEncoder":
        from sklearn.cluster import MiniBatchKMeans

        n_sample = int(n_sample_per_patch or self.sample_per_patch)
        sampled: list[np.ndarray] = []
        n_patches = 0
        for source in stream_factory():
            tokens = self._bb.patch_tokens(source)  # (T, C)
            if tokens is None or len(tokens) == 0:
                continue
            idx = np.random.choice(len(tokens), size=min(n_sample, len(tokens)), replace=False)
            sampled.append(tokens[idx])
            n_patches += 1

        if not sampled:
            raise ValueError("No DINO tokens provided for AnyLoc-VLAD training")

        all_tokens = np.vstack(sampled).astype(np.float32)
        logger.info(
            "dino_vlad_kmeans_start",
            n_tokens=len(all_tokens),
            n_centroids=self.n_centroids,
            n_patches=n_patches,
        )
        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_centroids,
            n_init=_s.global_vlad_kmeans_n_init,
            batch_size=min(10_000, len(all_tokens)),
            random_state=42,
            verbose=0,
        )
        self._kmeans.fit(all_tokens)
        logger.info("dino_vlad_fit_done", dim=self.dim)
        return self

    # ── inference ──────────────────────────────────────────────────────────
    def _vlad(self, tokens: np.ndarray) -> np.ndarray:
        if self._kmeans is None:
            raise RuntimeError("DinoVladEncoder not fitted. Call fit_images() or load() first.")
        embed_dim = self._bb.embed_dim
        if tokens is None or len(tokens) == 0:
            return np.zeros(self.n_centroids * embed_dim, dtype=np.float32)

        assignments = self._kmeans.predict(tokens)
        centroids = self._kmeans.cluster_centers_

        vlad = np.zeros((self.n_centroids, embed_dim), dtype=np.float32)
        for i in range(self.n_centroids):
            mask = assignments == i
            if np.any(mask):
                vlad[i] = (tokens[mask] - centroids[i]).sum(axis=0)

        # intra-normalization (per-centroid L2) — душит burstiness
        block_norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        vlad = vlad / (block_norms + 1e-12)

        v = vlad.flatten()
        v = np.sign(v) * np.sqrt(np.abs(v))  # power-law
        return _l2(v)

    def encode_image(self, source: ImageSource) -> np.ndarray:
        return self._vlad(self._bb.patch_tokens(source))

    def encode_image_batch(self, sources: Sequence[ImageSource]) -> np.ndarray:
        if not sources:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.encode_image(s) for s in sources]).astype(np.float32)

    def encode(self, descriptors: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError(
            "DinoVladEncoder работает с изображением: используй encode_image/encode_image_batch"
        )

    # ── serialization ──────────────────────────────────────────────────────
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or _s.global_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "type": "dino_vlad",
                    "model_name": self.model_name,
                    "image_size": self.image_size,
                    "n_centroids": self.n_centroids,
                    "sample_per_patch": self.sample_per_patch,
                    "kmeans": self._kmeans,
                },
                f,
            )
        logger.info("dino_vlad_saved", path=str(path), dim=self.dim)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DinoVladEncoder":
        path = Path(path or _s.global_model_path)
        if not path.exists():
            raise FileNotFoundError(f"DINO-VLAD encoder not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("type") != "dino_vlad":
            raise ValueError(f"{path} is not a DinoVladEncoder model (type={data.get('type')!r})")
        enc = cls(
            model_name=data["model_name"],
            image_size=data["image_size"],
            n_centroids=data["n_centroids"],
            sample_per_patch=data["sample_per_patch"],
        )
        enc._kmeans = data["kmeans"]
        logger.info("dino_vlad_loaded", path=str(path), dim=enc.dim)
        return enc
