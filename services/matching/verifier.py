"""
Верификация кандидатов: BFMatcher + ratio test + RANSAC affine (similarity).

Входные данные:
  - дескрипторы query-изображения
  - список кандидатов с их дескрипторами (загружаются из MinIO)

Выход: список кандидатов, отсортированных по score = inlier_count * inlier_ratio.
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
    photometric_score: float = 0.0  # NCC после варпа, 0.0 если проверка выключена/не считалась
    score: float = 0.0  # итоговый скор для сортировки (inliers * inlier_ratio * фактор NCC)


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


def _mutual_good_matches(
    bf: cv2.BFMatcher,  # type: ignore[name-defined]
    query_desc: np.ndarray,
    cand_desc: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:  # type: ignore[name-defined]
    """
    ПОПРОБОВАНО И ОТКЛЮЧЕНО (см. историю): Lowe ratio test в обе стороны +
    mutual (cross-check) фильтр — в теории должен резко снижать долю случайных
    совпадений на повторяющихся текстурах (ряды деревьев, поля, застройка).

    На практике на честном тесте (UAV-фото Харьковского ботсада, query
    ~2000 SIFT-точек vs Sentinel-патч 64×64px ~50-90 точек) mutual-check
    ПОЛНОСТЬЮ обнулил единственный рабочий сигнал: у query→cand было 6 good
    matches → 3 inliers, у mutual — 0 good matches → 0 inliers.

    Причина: knnMatch(cand, query) при N_query >> N_cand — ratio test в
    обратную сторону считается против богатого пула запроса, и почти все
    cand-дескрипторы получают два "похожих" соседа среди 2000 query-точек,
    так что ratio test проваливается почти для всех, даже для истинных
    соответствий. Mutual-check хорошо работает только при сопоставимых по
    размеру наборах дескрипторов — здесь это не так (богатый UAV-снимок
    против скудного мелкого спутникового патча). Функция оставлена в коде
    для справки/будущих экспериментов (например, если размер патчей вырастет
    и у них появится больше дескрипторов), но НЕ используется в проде.
    """
    matches_qc = bf.knnMatch(query_desc, cand_desc, k=2)
    good_qc = _ratio_test(matches_qc, ratio)
    if not good_qc:
        return []

    matches_cq = bf.knnMatch(cand_desc, query_desc, k=2)
    good_cq = _ratio_test(matches_cq, ratio)
    cq_pairs = {(m.trainIdx, m.queryIdx) for m in good_cq}  # (query_idx, cand_idx)

    return [m for m in good_qc if (m.queryIdx, m.trainIdx) in cq_pairs]


def _is_degenerate_affine(M: np.ndarray, min_scale: float = 1e-3, max_scale: float = 1e3) -> bool:
    """
    cv2.estimateAffinePartial2D(..., method=cv2.RANSAC) на малом числе точек
    (наш случай — единицы-десятки good matches на 64px патче) иногда находит
    вырожденное решение: линейная часть матрицы ~0, т.е. ВСЁ query-изображение
    схлопывается в одну точку кандидата. При достаточно мягком
    ransacReprojThreshold такое "решение" может набрать немало инлайеров
    (все точки кандидата, случайно оказавшиеся рядом с этой точкой), но оно
    физически бессмысленно — это не соответствие, а артефакт минимальной
    RANSAC-выборки (для similarity/partial-affine нужно всего 2 точки).

    Обнаружено при проверке photometric-фильтра: у truth-патча (см. честный
    тест) RANSAC дал 11 "инлайеров" с матрицей масштаба 0 — варп в точку,
    NCC корректно занулил такой "матч", что и вскрыло проблему. Без этой
    проверки бага не видно, т.к. inliers*inlier_ratio не смотрит на саму
    матрицу вообще.

    Считаем масштаб по норме первого столбца матрицы (для similarity-модели
    оба столбца имеют одинаковую норму = масштаб).
    """
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    return not (min_scale < scale < max_scale)


def _ransac_inliers(
    q_kp: list[cv2.KeyPoint],  # type: ignore[name-defined]
    c_kp: list[cv2.KeyPoint],  # type: ignore[name-defined]
    good_matches: list[cv2.DMatch],  # type: ignore[name-defined]
    ransac_threshold: float,
    min_good_matches: int | None = None,
) -> tuple[int, np.ndarray | None]:
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

    Вернуть (число инлайеров, affine-матрица 2×3 из query в candidate или None).
    Матрица нужна дальше для photometric-проверки (варп query → candidate).
    """
    min_matches = _s.min_good_matches if min_good_matches is None else min_good_matches
    if len(good_matches) < min_matches:
        return 0, None

    q_pts = np.float32([q_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    c_pts = np.float32([c_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(q_pts, c_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
    if mask is None or M is None or _is_degenerate_affine(M):
        return 0, None
    return int(mask.sum()), M


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-normalized cross-correlation между двумя наборами пикселей одинаковой формы."""
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-6:
        return 0.0
    return float((a * b).sum() / denom)


def _photometric_score(
    query_gray: np.ndarray,
    cand_gray: np.ndarray,
    affine_query_to_cand: np.ndarray,
    min_overlap_ratio: float,
) -> float:
    """
    Независимая фотометрическая проверка ПОВЕРХ геометрического RANSAC-скора.

    RANSAC/inlier_count говорит только "согласуются ли точки геометрически" —
    это не отличает настоящее совпадение от случайного на повторяющейся
    текстуре (ряды деревьев, поля, регулярная застройка), где легко набрать
    геометрически консистентный набор точек у совершенно другого участка.

    Здесь варпим query-изображение (обычно значительно больше кандидата) в
    систему координат кандидата через affine-матрицу из RANSAC и считаем NCC
    на пересечении — если это реально один и тот же участок земли, пиксели
    после совмещения должны быть похожи (с поправкой на сезон/сенсор/шум),
    если совпадение случайное — корреляция будет околонулевой или отрицательной.

    Варпим именно query → cand (а не наоборот), т.к. кандидат маленький
    (64×64px при текущем patch_size), и апскейлить его до размера query не
    даёт новой информации, только интерполяционные артефакты — дешевле и
    честнее сравнивать в нативном разрешении патча.

    Возвращает NCC в диапазоне [0, 1] (отрицательная корреляция обрезается
    до 0 — это явно "не то же самое место", а не слабый, но валидный сигнал).
    0.0 также возвращается, если после варпа с query в кадр кандидата попало
    слишком мало валидных пикселей (несогласованный масштаб/сильно неточная
    матрица) — недостаточно данных для честной оценки.
    """
    h, w = cand_gray.shape[:2]

    warped = cv2.warpAffine(
        query_gray, affine_query_to_cand, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # Компаньон-маска: какие пиксели кандидата реально накрыты проекцией query
    # (а не попали за его пределы / в borderValue-заливку).
    src_mask = np.full(query_gray.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpAffine(
        src_mask, affine_query_to_cand, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = warped_mask > 200
    overlap_ratio = float(valid.sum()) / (h * w)
    if overlap_ratio < min_overlap_ratio:
        return 0.0

    return max(0.0, _ncc(warped[valid], cand_gray[valid]))


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
        query_gray: np.ndarray | None = None,
    ) -> list[VerifiedCandidate]:
        """
        candidates: список словарей {patch_id, center_lat, center_lon, bbox, s3_path}
        load_desc_fn: callable(s3_path) → (keypoints, descriptors, gray) | (None, None, None)
            gray может быть None — тогда photometric-проверка просто пропускается
            для этого кандидата (нейтральный фактор 1.0).
        query_gray: grayscale query ПОСЛЕ той же предобработки, что видел SIFT
            (см. `services.features.sift.extract_query_gray`). Нужен только
            для photometric-проверки; если None — проверка пропускается совсем
            (обратная совместимость со старыми вызовами/тестами).

        Returns: список VerifiedCandidate, sorted by score DESC
            (score = inliers * inlier_ratio * photometric_factor).
        """
        if query_desc is None or len(query_desc) == 0:
            logger.warning("verifier_no_query_desc")
            return []

        do_photometric = _s.photometric_check_enabled and query_gray is not None

        verified: list[VerifiedCandidate] = []

        for cand in candidates:
            try:
                cand_kp, cand_desc, cand_gray = load_desc_fn(cand["s3_path"])
            except Exception as exc:
                logger.warning("verifier_load_error", s3_path=cand["s3_path"], error=str(exc))
                continue

            if cand_desc is None or len(cand_desc) < _s.min_good_matches:
                continue

            # Односторонний ratio test (не mutual — см. docstring _mutual_good_matches:
            # cross-check ломает матчинг при сильно асимметричных наборах дескрипторов,
            # как в нашем случае "богатый UAV-query vs скудный 64px патч").
            matches = self._bf.knnMatch(query_desc, cand_desc, k=2)
            good = _ratio_test(matches, self.lowe_ratio)

            inliers, M = _ransac_inliers(query_kp, cand_kp, good, self.ransac_threshold)

            if inliers > 0:
                # inlier_ratio = доля good-матчей, подтверждённых RANSAC.
                # Одного inlier_count недостаточно: тайлы с "богатой" текстурой
                # (застройка) статистически дают больше keypoints/матчей и,
                # соответственно, больше инлаеров просто по объёму, независимо
                # от того, верный это тайл или нет. inlier_ratio нормирует
                # это смещение.
                inlier_ratio = inliers / len(good) if good else 0.0

                # Photometric-фактор: независимая проверка "похожи ли пиксели
                # после совмещения", а не только "согласуются ли точки
                # геометрически" (см. docstring _photometric_score). Нейтральный
                # фактор 1.0, если проверка выключена/недоступна для кандидата —
                # тогда поведение идентично старому score = inliers * inlier_ratio.
                photometric = 1.0
                if do_photometric and cand_gray is not None and M is not None:
                    photometric = _photometric_score(
                        query_gray, cand_gray, M, _s.photometric_min_overlap_ratio
                    )

                score = inliers * inlier_ratio * photometric
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
                        photometric_score=round(photometric, 4),
                        score=round(score, 4),
                    )
                )

        verified.sort(key=lambda x: x.score, reverse=True)
        logger.info(
            "verifier_done",
            n_candidates=len(candidates),
            n_verified=len(verified),
            photometric_enabled=do_photometric,
        )
        return verified[: self.top_n]
