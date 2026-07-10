"""
Фабрика coarse-энкодера (глобальный дескриптор патча для FAISS-поиска).

Единая точка выбора метода приближённого поиска. Все энкодеры реализуют один
протокол:

    encode(descriptors) -> np.ndarray  # (dim,), L2-нормированный
    dim                  -> int
    is_fitted            -> bool
    save(path) / load(path)

Это позволяет менять coarse-стадию (bovw → vlad → в будущем ASMK/DINOv2), не
трогая ни verifier (SIFT+RANSAC), ни localize/index_task — они работают через
эту фабрику. Метод выбирается настройкой COARSE_METHOD.
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


@runtime_checkable
class CoarseEncoder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def is_fitted(self) -> bool: ...

    def encode(self, descriptors: np.ndarray | None) -> np.ndarray: ...

    def save(self, path: Path | None = None) -> Path: ...


def coarse_model_path(method: str | None = None) -> Path:
    method = (method or _s.coarse_method).lower()
    return Path(_s.vlad_path if method == "vlad" else _s.vocab_path)


def load_coarse_encoder(method: str | None = None) -> CoarseEncoder:
    """Загрузить обученный coarse-энкодер согласно COARSE_METHOD (или явному method)."""
    method = (method or _s.coarse_method).lower()
    if method == "vlad":
        return VladEncoder.load()
    if method == "bovw":
        return Vocabulary.load()
    raise ValueError(f"Unknown coarse_method: {method!r} (expected 'bovw' or 'vlad')")


def new_coarse_encoder(method: str | None = None) -> CoarseEncoder:
    """Создать НЕобученный coarse-энкодер согласно COARSE_METHOD (для build-задач)."""
    method = (method or _s.coarse_method).lower()
    if method == "vlad":
        return VladEncoder()
    if method == "bovw":
        return Vocabulary()
    raise ValueError(f"Unknown coarse_method: {method!r} (expected 'bovw' or 'vlad')")
