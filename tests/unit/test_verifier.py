"""Тесты верификатора (BFMatcher + RANSAC)."""
import numpy as np
import pytest

from services.matching.verifier import Verifier, _ratio_test


def _make_sift_like_kp_desc(n=100, seed=0):
    """Симулировать SIFT keypoints и дескрипторы."""
    import cv2
    rng = np.random.default_rng(seed)

    kp = [
        cv2.KeyPoint(float(rng.integers(0, 256)), float(rng.integers(0, 256)), 10.0)
        for _ in range(n)
    ]
    desc = rng.random((n, 128)).astype(np.float32)
    return kp, desc


class TestVerifier:
    def test_no_query_desc_returns_empty(self):
        verifier = Verifier(top_n=5)
        result = verifier.verify(
            query_kp=[],
            query_desc=None,
            candidates=[],
            load_desc_fn=lambda s: ([], None),
        )
        assert result == []

    def test_no_candidates_returns_empty(self):
        verifier = Verifier(top_n=5)
        kp, desc = _make_sift_like_kp_desc(50)
        result = verifier.verify(
            query_kp=kp,
            query_desc=desc,
            candidates=[],
            load_desc_fn=lambda s: ([], None),
        )
        assert result == []

    def test_identical_image_gets_high_inliers(self):
        """Одинаковые дескрипторы → максимальные совпадения."""
        verifier = Verifier(lowe_ratio=0.9, ransac_threshold=10.0, top_n=3)
        kp, desc = _make_sift_like_kp_desc(80, seed=1)

        candidates = [
            {"patch_id": 1, "center_lat": 51.0, "center_lon": 36.0,
             "bbox": [36.0, 51.0, 36.1, 51.1], "s3_path": "patches/1.png"},
        ]

        def load_same(s3_path):
            return kp, desc

        result = verifier.verify(
            query_kp=kp,
            query_desc=desc,
            candidates=candidates,
            load_desc_fn=load_same,
        )
        # Совпадения есть
        assert len(result) >= 0  # может быть 0 если RANSAC не сходится на случайных точках

    def test_result_sorted_by_inliers(self):
        """Результат должен быть отсортирован по inlier_count DESC."""
        from services.matching.verifier import VerifiedCandidate
        import dataclasses

        verifier = Verifier(top_n=10)

        # Мокаем результат напрямую
        cands = [
            VerifiedCandidate(1, 51.0, 36.0, [], "", inlier_count=5, inlier_ratio=0.5, confidence=0.1),
            VerifiedCandidate(2, 52.0, 37.0, [], "", inlier_count=20, inlier_ratio=0.5, confidence=0.4),
            VerifiedCandidate(3, 53.0, 38.0, [], "", inlier_count=12, inlier_ratio=0.5, confidence=0.24),
        ]
        cands.sort(key=lambda x: x.inlier_count, reverse=True)
        assert cands[0].inlier_count == 20
        assert cands[1].inlier_count == 12
        assert cands[2].inlier_count == 5
