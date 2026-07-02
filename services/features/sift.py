"""SIFT: извлечение ключевых точек и дескрипторов из изображения."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import get_logger, get_settings
from services.features.preprocess import preprocess_for_matching

logger = get_logger(__name__)
_s = get_settings()


def _build_detector() -> cv2.SIFT:  # type: ignore[name-defined]
    return cv2.SIFT_create(
        nfeatures=_s.sift_n_features,
        contrastThreshold=_s.sift_contrast_threshold,
        edgeThreshold=_s.sift_edge_threshold,
        sigma=_s.sift_sigma,
    )


def load_image(source: Path | bytes | np.ndarray) -> np.ndarray:
    """Загрузить изображение как есть (BGR или grayscale), без конвертации."""
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Cannot load image from {type(source)}")
    return img


def load_image_gray(source: Path | bytes | np.ndarray) -> np.ndarray:
    """Загрузить изображение в grayscale uint8."""
    img = load_image(source)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray


def extract_descriptors(
    source: Path | bytes | np.ndarray,
    max_side: int = 1024,
    resize_scale: float | None = None,
    normalize: bool | None = None,
    use_clahe: bool | None = None,
    use_lcn: bool | None = None,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:  # type: ignore[name-defined]
    """
    Извлечь SIFT дескрипторы из изображения.

    Параметры предобработки (resize_scale/normalize/use_clahe/use_lcn) по
    умолчанию берутся из настроек (`config.settings`) — см. `preprocess.py`.
    Это гарантирует, что и патчи базы (index_task), и query-изображение
    (localize) проходят ОДИНАКОВУЮ предобработку, что критично для честного
    сравнения между разнородными доменами (UAV vs Sentinel-2).

    Возвращает (keypoints, descriptors).
    descriptors: float32 array shape (N, 128) или None если точек нет.
    """
    resize_scale = _s.preprocess_resize_scale if resize_scale is None else resize_scale
    normalize = _s.preprocess_normalize_channels if normalize is None else normalize
    use_clahe = _s.preprocess_use_clahe if use_clahe is None else use_clahe
    use_lcn = _s.preprocess_use_lcn if use_lcn is None else use_lcn

    img = load_image(source)
    gray = preprocess_for_matching(
        img,
        resize_scale=resize_scale,
        normalize=normalize,
        use_clahe=use_clahe,
        use_lcn=use_lcn,
    )

    # Ограничиваем размер для скорости (не меняет координаты ключевых точек)
    h, w = gray.shape
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    detector = _build_detector()
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) == 0:
        logger.debug("sift_no_descriptors", size=(w, h))
        return [], None

    logger.debug("sift_extracted", n_keypoints=len(keypoints), shape=gray.shape)
    return list(keypoints), descriptors.astype(np.float32)
