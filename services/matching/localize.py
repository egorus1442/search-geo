"""
Полный online pipeline: изображение → top-N гео-кандидатов.

Шаги:
  1. SIFT extraction из query-изображения
  2. BoVW encoding → histogram
  3. FAISS search → top-100 coarse candidates (только patch_id)
  4. Загрузка метаданных из PostgreSQL (координаты, s3_path)
  5. Верификация (BFMatcher + RANSAC) → top-N
  6. Возврат результатов

Если Settings.exhaustive_search=True, шаги 2-3 (BoVW/FAISS) пропускаются и
на верификацию (шаг 5) подаются ВСЕ патчи из БД напрямую. Временный режим
для маленькой базы, пока BoVW coarse-фильтр плохо откалиброван (см. честный
тест на UAV-фото Харьковского ботсада — BoVW отсекал верный патч ещё до
RANSAC). Включается через .env: EXHAUSTIVE_SEARCH=true.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from config import get_logger, get_settings
from services.features.sift import (
    extract_patch_descriptors,
    extract_patch_gray,
    extract_query_descriptors,
    extract_query_gray,
)
from services.features.coarse import (
    coarse_index_path,
    encoder_input_kind,
    load_coarse_encoder,
)
from services.index.faiss_store import FaissStore
from services.index.metadata_store import PatchRepo
from services.ingestor.storage import download_bytes
from services.matching.verifier import Verifier, VerifiedCandidate
from services.db.session import SyncSessionLocal

logger = get_logger(__name__)
_s = get_settings()


@lru_cache(maxsize=1)
def _get_vocab():
    """Обученный coarse-энкодер (BoVW или VLAD, см. COARSE_METHOD)."""
    return load_coarse_encoder()


@lru_cache(maxsize=1)
def _get_faiss() -> FaissStore:
    # Нейро-методы (dino/dino_vlad) держат отдельный индекс (GLOBAL_INDEX_PATH).
    return FaissStore.load(path=coarse_index_path())


@lru_cache(maxsize=1)
def _get_verifier() -> Verifier:
    return Verifier()


def _load_candidate_descriptors(s3_path: str):
    """
    Скачать патч из MinIO и извлечь SIFT дескрипторы + grayscale (для
    photometric-проверки после варпа, см. verifier.py). Патч уже маленький
    (64×64px при текущем patch_size), так что держать ещё и gray в памяти
    рядом с дескрипторами дёшево.
    """
    img_bytes = download_bytes(s3_path)
    kp, desc = extract_patch_descriptors(img_bytes)
    gray = extract_patch_gray(img_bytes) if _s.photometric_check_enabled else None
    return kp, desc, gray


def localize(image_bytes: bytes, top_n: int | None = None) -> list[dict[str, Any]]:
    """
    Главная функция геолокализации.

    image_bytes: сырые байты изображения (JPEG/PNG)
    top_n: количество возвращаемых кандидатов

    Returns список словарей с полями:
        rank, patch_id, center_lat, center_lon, bbox,
        inlier_count, confidence, thumbnail_url
    """
    top_n = top_n or _s.top_n_result

    # ── Step 1: SIFT ──────────────────────────────────────────────────────────
    query_kp, query_desc = extract_query_descriptors(image_bytes)
    if query_desc is None:
        logger.warning("localize_no_features")
        return []

    query_gray = extract_query_gray(image_bytes) if _s.photometric_check_enabled else None

    if _s.exhaustive_search:
        # ── Steps 2-4 (bypass): без BoVW/FAISS, берём ВСЕ патчи из БД ─────────
        # Временный режим, см. Settings.exhaustive_search.
        logger.info("localize_exhaustive_mode")
        with SyncSessionLocal() as session:
            repo = PatchRepo(session)
            all_ids = repo.get_all_patch_ids()
            candidates = repo.get_patches_by_ids(all_ids)

        if not candidates:
            logger.warning("localize_no_candidates")
            return []
    else:
        # ── Step 2: coarse encoding (COARSE_METHOD) ───────────────────────────
        vocab = _get_vocab()
        if encoder_input_kind(vocab) == "image":
            # dino/dino_vlad: эмбеддинг «места» из САМОЙ картинки query (без SIFT
            # и без query-resize 0.06 — DINOv2 сам ресайзит до GLOBAL_IMAGE_SIZE).
            query_hist = vocab.encode_image(image_bytes)  # type: ignore[attr-defined]
            coarse_k = _s.global_top_k
        else:
            query_hist = vocab.encode(query_desc)
            coarse_k = _s.top_n_coarse

        # ── Step 3: FAISS coarse search ────────────────────────────────────────
        store = _get_faiss()
        distances, candidate_ids = store.search(query_hist, k=coarse_k)

        # Фильтруем невалидные ID (-1 у FAISS означает пустую ячейку)
        valid_ids = [int(pid) for pid in candidate_ids if pid != -1]
        if not valid_ids:
            logger.warning("localize_no_faiss_candidates")
            return []

        # ── Step 4: Metadata from PostgreSQL ─────────────────────────────────────
        with SyncSessionLocal() as session:
            repo = PatchRepo(session)
            candidates = repo.get_patches_by_ids(valid_ids)

        if not candidates:
            return []

    # ── Step 5: RANSAC verification ───────────────────────────────────────────
    verifier = _get_verifier()
    verified = verifier.verify(
        query_kp=query_kp,
        query_desc=query_desc,
        candidates=candidates,
        load_desc_fn=_load_candidate_descriptors,
        query_gray=query_gray,
    )

    # ── Step 6: Format result ─────────────────────────────────────────────────
    results = []
    for rank, cand in enumerate(verified[:top_n], start=1):
        try:
            thumbnail_url = _get_thumbnail_url(cand.s3_path)
        except Exception:
            thumbnail_url = None

        results.append({
            "rank": rank,
            "patch_id": cand.patch_id,
            "center_lat": round(cand.center_lat, 6),
            "center_lon": round(cand.center_lon, 6),
            "bbox": [round(c, 6) for c in cand.bbox],
            "inlier_count": cand.inlier_count,
            "inlier_ratio": cand.inlier_ratio,
            "photometric_score": cand.photometric_score,
            "confidence": cand.confidence,
            "thumbnail_url": thumbnail_url,
        })

    logger.info("localize_done", n_results=len(results))
    return results


def _get_thumbnail_url(s3_path: str) -> str | None:
    from services.ingestor.storage import get_presigned_url
    try:
        return get_presigned_url(s3_path, expires_seconds=3600)
    except Exception:
        return None


def reload_indexes() -> None:
    """Принудительно перезагрузить словарь и FAISS из файлов."""
    _get_vocab.cache_clear()
    _get_faiss.cache_clear()
    logger.info("indexes_reloaded")
