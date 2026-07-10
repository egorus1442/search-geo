"""
RootSIFT — Hellinger-нормализация SIFT-дескрипторов (Arandjelović & Zisserman, 2012).

Преобразование: L1-нормировать каждый дескриптор → взять поэлементный sqrt.
После этого евклидово расстояние между RootSIFT-дескрипторами эквивалентно
Hellinger-расстоянию между исходными (гистограммными) SIFT — это стабильно
повышает различимость на 2-4% в любой SIFT-задаче практически бесплатно.

ВАЖНО: применяется ТОЛЬКО в retrieval-пути (VLAD-агрегация). Дескрипторы,
уходящие в verifier (SIFT + BFMatcher + RANSAC), остаются сырыми — чтобы не
менять проверенное поведение геометрической верификации.
"""
from __future__ import annotations

import numpy as np


def to_rootsift(descriptors: np.ndarray | None, eps: float = 1e-7) -> np.ndarray | None:
    """
    SIFT-дескрипторы (N, 128) → RootSIFT (N, 128), float32.

    None/пустой вход возвращается как есть (None).
    """
    if descriptors is None or len(descriptors) == 0:
        return descriptors

    d = descriptors.astype(np.float32, copy=True)
    d /= (d.sum(axis=1, keepdims=True) + eps)  # L1
    np.sqrt(d, out=d)
    return d
