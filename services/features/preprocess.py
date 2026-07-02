"""
Предобработка изображений перед SIFT.

Идея: UAV-снимок и Sentinel-2 патч технически один и тот же участок земли,
но выглядят по-разному — разный GSD (масштаб), разный цветовой баланс/уровни,
разное освещение/сезон/атмосфера. SIFT+BoVW чувствителен ко всему этому.

Здесь собраны классические для обработки геоизображений приёмы, призванные
сократить этот разрыв ДО извлечения признаков:

  - resize_by_scale / resize_to_gsd  — согласование масштаба (понижение
    разрешения UAV-снимка ближе к GSD базы)
  - normalize_channels               — percentile-стретч по каждому каналу
    (устраняет разницу цветового баланса/уровней между сенсорами)
  - equalize_local_contrast          — CLAHE, адаптивное выравнивание контраста
  - local_contrast_normalization_map — "нормализованная карта" (LCN):
    (I - локальное_среднее) / локальное_std — убирает влияние абсолютной
    яркости и локальной освещённости, оставляя только относительный контраст.
    Классический приём в обработке геоизображений для детекции объектов
    и сопоставления разнородных снимков.

Все функции — чистые (numpy in/out), без побочных эффектов, чтобы их было
удобно дёргать по отдельности в экспериментах (scripts/experiment_preprocessing.py).
"""
from __future__ import annotations

import cv2
import numpy as np

from config import get_logger

logger = get_logger(__name__)


def resize_by_scale(img: np.ndarray, scale: float) -> np.ndarray:
    """Изменить размер изображения в `scale` раз (scale < 1 — понижение разрешения)."""
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def resize_to_gsd(img: np.ndarray, source_gsd_m: float, target_gsd_m: float) -> np.ndarray:
    """
    Согласовать масштаб с эталонной базой: если снимок имеет GSD `source_gsd_m`
    м/пкс, а база построена с GSD `target_gsd_m` — пересчитать так, чтобы
    один пиксель соответствовал одинаковому размеру на земле в обоих случаях.

    Пример: UAV снят с GSD ~0.1 м/пкс, база — Sentinel-2 10 м/пкс →
    scale = 0.1 / 10 = 0.01 (сильное понижение разрешения UAV-снимка).
    """
    scale = source_gsd_m / target_gsd_m
    return resize_by_scale(img, scale)


def normalize_channels(img: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """
    Percentile-стретч по каждому каналу отдельно (та же логика, что и при
    нарезке Sentinel-патчей в `tile_cutter._normalize_band`) — приводит
    UAV-снимок и Sentinel-патч к сопоставимому диапазону уровней/цветового
    баланса вне зависимости от исходной радиометрии сенсора.
    """
    if img.ndim == 2:
        channels = [img]
    else:
        channels = [img[..., c] for c in range(img.shape[-1])]

    normalized = []
    for ch in channels:
        valid = ch[ch > 0]
        if valid.size == 0:
            normalized.append(np.zeros_like(ch, dtype=np.uint8))
            continue
        lo, hi = np.percentile(valid, (low_pct, high_pct))
        if hi <= lo:
            normalized.append(np.full_like(ch, 128, dtype=np.uint8))
            continue
        clipped = np.clip(ch, lo, hi).astype(np.float32)
        normalized.append(((clipped - lo) / (hi - lo) * 255).astype(np.uint8))

    return normalized[0] if img.ndim == 2 else np.stack(normalized, axis=-1)


def equalize_local_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """CLAHE — адаптивное выравнивание контраста по локальным окнам."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray)


def local_contrast_normalization_map(
    gray: np.ndarray,
    sigma: float = 8.0,
    eps: float = 1e-3,
    clip_sigmas: float = 3.0,
) -> np.ndarray:
    """
    "Нормализованная карта" (Local Contrast Normalization, LCN):

        map = (I - local_mean) / (local_std + eps)

    Классический приём предобработки геоизображений перед детекцией
    объектов/признаков. Убирает влияние абсолютного уровня яркости и
    локальной освещённости (тени, засветка, атмосферная дымка), оставляя
    только относительный контраст текстуры — то, что реально отличает
    один объект/участок местности от другого. За счёт этого представление
    гораздо устойчивее к различию радиометрии между UAV-камерой и
    Sentinel-2, сезонным сдвигам, разнице в экспозиции и т.п.

    Возвращает uint8 карту, пригодную для дальнейшего SIFT.
    """
    f = gray.astype(np.float32)
    local_mean = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    local_sq_mean = cv2.GaussianBlur(f * f, (0, 0), sigmaX=sigma)
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)
    local_std = np.sqrt(local_var)

    normalized = (f - local_mean) / (local_std + eps)
    normalized = np.clip(normalized, -clip_sigmas, clip_sigmas)
    normalized = ((normalized + clip_sigmas) / (2 * clip_sigmas) * 255).astype(np.uint8)
    return normalized


def preprocess_for_matching(
    img: np.ndarray,
    resize_scale: float | None = None,
    normalize: bool = False,
    use_clahe: bool = False,
    use_lcn: bool = False,
) -> np.ndarray:
    """
    Единый пайплайн предобработки перед SIFT.

    ВАЖНО: применяется одинаково и к query (UAV), и к патчам базы (Sentinel)
    — иначе сравнение нечестное. При вызове через `extract_descriptors`
    (см. services/features/sift.py) это гарантируется автоматически, т.к.
    параметры берутся из общих настроек (config.settings) для обоих путей.

    Порядок шагов важен:
      1. resize   — сначала согласовать масштаб (до нормализации уровней,
                    т.к. percentile-статистика чувствительна к масштабу окна)
      2. normalize channels — устранить разницу цветового баланса
      3. → grayscale
      4. clahe и/или lcn — локальные приёмы контраста (применяются к серому)
    """
    out = img

    if resize_scale is not None and resize_scale != 1.0:
        out = resize_by_scale(out, resize_scale)

    if normalize:
        out = normalize_channels(out)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out

    if use_clahe:
        gray = equalize_local_contrast(gray)

    if use_lcn:
        gray = local_contrast_normalization_map(gray)

    return gray
