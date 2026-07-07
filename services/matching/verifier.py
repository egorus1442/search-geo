"""
Верификация кандидатов: BFMatcher + ratio test + RANSAC homography.

Входные данные:
  - дескрипторы query-изображения
  - список кандидатов с их дескрипторами (загружаются из MinIO)

Выход: список кандидатов, отсортированных по числу RANSAC-инлайеров.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from config import get_logger, get_settings

logger = get_logger(__name__)
_s = get_settings()


@dataclass
class VerifiedCandidate:
    patch_id: int
    center_lat: float
    center_lon: float
    bbox: list[float]
    s3_path: str
    inlier_count: int
    inlier_ratio: float
    confidence: float


def _ratio_test(
    matches: list[list[cv2.DMatch]],  # type: ignore[name-defined]
    ratio: float,
) -> list[cv2.DMatch]:  # type: ignore[name-defined]
    """Lowe's ratio test."""
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def _ransac_inliers(
    q_kp: list[cv2.KeyPoint],  # type: ignore[name-defined]
    c_kp: list[cv2.KeyPoint],  # type: ignore[name-defined]
    good_matches: list[cv2.DMatch],  # type: ignore[name-defined]
    ransac_threshold: float,
    min_good_matches: int | None = None,
) -> int:
    """
    Применить RANSAC для нахождения геометрического преобразования между
    query (надирный UAV-снимок) и кандидатом (ортопатч Sentinel-2).

    Используем ограниченную модель — similarity/partial-affine (поворот +
    единый масштаб + сдвиг, 4 DoF) вместо полной homography (8 DoF).
    Физически оба снимка близки к надирным ортопроекциям одного участка
    земли, так что реальное преобразование между ними — почти affine, а не
    произвольная перспектива. Homography с 8 степенями свободы слишком легко
    "натягивается" на случайный набор совпадений при повторяющихся текстурах
    (поля, лес, ряды застройки), давая ложно высокий inlier_count у неверных
    тайлов. Affine-модель ограничивает вырожденные решения и должна резко
    снизить долю таких ложных срабатываний.

    Вернуть число инлайеров.
    """
    min_matches = _s.min_good_matches if min_good_matches is None else min_good_matches
    if len(good_matches) < min_matches:
        return 0

    q_pts = np.float32([q_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    c_pts = np.float32([c_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.estimateAffinePartial2D(q_pts, c_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
    if mask is None:
        return 0
    return int(mask.sum())


class Verifier:
    """
    Верифицирует coarse-кандидатов локальным сопоставлением SIFT + RANSAC.

    Пример использования:
        verifier = Verifier()
        results = verifier.verify(
            query_kp=q_kp,
            query_desc=q_desc,
            candidates=candidate_meta_list,   # из PostgreSQL
            load_desc_fn=lambda s3_key: ...,  # функция загрузки из MinIO
        )
    """

    def __init__(
        self,
        lowe_ratio: float | None = None,
        ransac_threshold: float | None = None,
        top_n: int | None = None,
    ) -> None:
        self.lowe_ratio = lowe_ratio or _s.lowe_ratio
        self.ransac_threshold = ransac_threshold or _s.ransac_threshold
        self.top_n = top_n or _s.top_n_result

        self._bf = cv2.BFMatcher(cv2.NORM_L2)

    def verify(
        self,
        query_kp: list[cv2.KeyPoint],  # type: ignore[name-defined]
        query_desc: np.ndarray,
        candidates: list[dict[str, Any]],
        load_desc_fn,
    ) -> list[VerifiedCandidate]:
        """
        candidates: список словарей {patch_id, center_lat, center_lon, bbox, s3_path}
        load_desc_fn: callable(s3_path) → (keypoints, descriptors) | (None, None)

        Returns: список VerifiedCandidate, sorted by inlier_count DESC.
        """
        if query_desc is None or len(query_desc) == 0:
            logger.warning("verifier_no_query_desc")
            return []

        verified: list[VerifiedCandidate] = []

        for cand in candidates:
            try:
                cand_kp, cand_desc = load_desc_fn(cand["s3_path"])
            except Exception as exc:
                logger.warning("verifier_load_error", s3_path=cand["s3_path"], error=str(exc))
                continue

            if cand_desc is None or len(cand_desc) < _s.min_good_matches:
                continue

            matches = self._bf.knnMatch(query_desc, cand_desc, k=2)
            good = _ratio_test(matches, self.lowe_ratio)

            inliers = _ransac_inliers(query_kp, cand_kp, good, self.ransac_threshold)

            if inliers > 0:
                # inlier_ratio = доля good-матчей, подтверждённых RANSAC.
                # Одного inlier_count недостаточно: тайлы с "богатой" текстурой
                # (застройка) статистически дают больше keypoints/матчей и,
                # соответственно, больше инлаеров просто по объёму, независимо
                # от того, верный это тайл или нет. inlier_ratio нормирует
                # это смещение. Итоговый score = inliers * inlier_ratio —
                # компромисс между "достаточно доказательств" (count) и
                # "насколько чистое совпадение" (ratio), см. рекомендацию
                # по борьбе с ложными срабатываниями на похожих тайлах.
                inlier_ratio = inliers / len(good) if good else 0.0
                confidence = min(inliers / 50.0, 1.0)
                verified.append(
                    VerifiedCandidate(
                        patch_id=cand["patch_id"],
                        center_lat=cand["center_lat"],
                        center_lon=cand["center_lon"],
                        bbox=cand["bbox"],
                        s3_path=cand["s3_path"],
                        inlier_count=inliers,
                        inlier_ratio=round(inlier_ratio, 4),
                        confidence=round(confidence, 4),
                    )
                )

        verified.sort(key=lambda x: x.inlier_count * x.inlier_ratio, reverse=True)
        logger.info(
            "verifier_done",
            n_candidates=len(candidates),
            n_verified=len(verified),
        )
        return verified[: self.top_n]
