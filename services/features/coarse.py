"""
Фабрика coarse-энкодера (глобальный дескриптор патча для FAISS-поиска).

Единая точка выбора метода приближённого поиска. Все энкодеры реализуют один
протокол:

    encode(descriptors) -> np.ndarray  # (dim,), L2-нормированный
    dim                  -> int
    is_fitted            -> bool
    save(path) / load(path)

Есть два вида энкодеров, различаемых по input_kind:
  - descriptors (vlad/bovw) — кодируют SIFT-дескрипторы через encode(descs);
  - image (dino/dino_vlad)  — кодируют САМО изображение через
    encode_image(source)/encode_image_batch(sources), input_kind="image".
Consumers (index_task/localize/eval_recall) диспетчеризуют по encoder_input_kind.

Это позволяет менять coarse-стадию (bovw → vlad → dino/dino_vlad(AnyLoc)), не
трогая ни verifier (SIFT+RANSAC), ни verifier-путь — они работают через эту
фабрику. Метод выбирается настройкой COARSE_METHOD.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from config import get_logger, get_settings
from services.features.vlad import VladEncoder
from services.features.vocabulary import Vocabulary

logger = get_logger(__name__)
_s = get_settings()


# Методы, кодирующие из САМОЙ картинки (а не из SIFT-дескрипторов). Для них
# энкодер имеет input_kind="image" и методы encode_image/encode_image_batch;
# index_task/localize диспетчеризуют по input_kind (см. хелперы ниже).
IMAGE_METHODS = frozenset({"dino", "dino_vlad"})
_KNOWN_METHODS = frozenset({"bovw", "vlad"}) | IMAGE_METHODS


@runtime_checkable
class CoarseEncoder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def is_fitted(self) -> bool: ...

    def encode(self, descriptors: np.ndarray | None) -> np.ndarray: ...

    def save(self, path: Path | None = None) -> Path: ...


def is_image_method(method: str | None = None) -> bool:
    """True для нейро-методов (dino/dino_vlad), кодирующих из изображения."""
    return (method or _s.coarse_method).lower() in IMAGE_METHODS


def encoder_input_kind(encoder: object) -> str:
    """'image' | 'descriptors'. VLAD/BoVW не имеют атрибута → 'descriptors'."""
    return getattr(encoder, "input_kind", "descriptors")


def coarse_model_path(method: str | None = None) -> Path:
    """Путь к сохранённой модели энкодера для метода."""
    method = (method or _s.coarse_method).lower()
    if method in IMAGE_METHODS:
        return Path(_s.global_model_path)
    return Path(_s.vlad_path if method == "vlad" else _s.vocab_path)


def coarse_index_path(method: str | None = None) -> Path:
    """
    Путь к FAISS-индексу для метода. Нейро-методы держат ОТДЕЛЬНЫЙ индекс
    (GLOBAL_INDEX_PATH), чтобы vlad/bovw и dino/dino_vlad не затирали друг друга
    и можно было переключаться через .env без пересборки каждый раз.
    """
    method = (method or _s.coarse_method).lower()
    return Path(_s.global_index_path if method in IMAGE_METHODS else _s.faiss_index_path)


def load_coarse_encoder(method: str | None = None) -> CoarseEncoder:
    """Загрузить обученный coarse-энкодер согласно COARSE_METHOD (или явному method)."""
    method = (method or _s.coarse_method).lower()
    if method == "vlad":
        return VladEncoder.load()
    if method == "bovw":
        return Vocabulary.load()
    if method == "dino":
        from services.features.dino import DinoEncoder  # ленивый импорт torch/timm
        return DinoEncoder.load()
    if method == "dino_vlad":
        from services.features.dino import DinoVladEncoder
        return DinoVladEncoder.load()
    raise ValueError(f"Unknown coarse_method: {method!r} (expected one of {sorted(_KNOWN_METHODS)})")


def new_coarse_encoder(method: str | None = None) -> CoarseEncoder:
    """Создать НЕобученный coarse-энкодер согласно COARSE_METHOD (для build-задач)."""
    method = (method or _s.coarse_method).lower()
    if method == "vlad":
        return VladEncoder()
    if method == "bovw":
        return Vocabulary()
    if method == "dino":
        from services.features.dino import DinoEncoder
        return DinoEncoder()
    if method == "dino_vlad":
        from services.features.dino import DinoVladEncoder
        return DinoVladEncoder()
    raise ValueError(f"Unknown coarse_method: {method!r} (expected one of {sorted(_KNOWN_METHODS)})")
