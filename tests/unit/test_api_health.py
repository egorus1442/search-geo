"""Тесты API эндпоинтов (без реальных зависимостей — через моки)."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Health endpoint должен отвечать даже если зависимости недоступны."""
    with (
        patch("services.api.routes.health.AsyncSessionLocal") as mock_db,
        patch("services.api.routes.health.redis") as mock_redis,
        patch("services.api.routes.health.Minio") as mock_minio,
        patch("services.api.routes.health._get_faiss") as mock_faiss,
    ):
        mock_db.return_value.__aenter__ = MagicMock(return_value=MagicMock())
        mock_redis.from_url.return_value.ping.return_value = True
        mock_minio.return_value.bucket_exists.return_value = True
        mock_faiss.return_value.ntotal = 1000

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "components" in data


def test_localize_no_file_returns_422():
    """Запрос без файла → 422 Unprocessable Entity."""
    resp = client.post("/api/v1/localize")
    assert resp.status_code == 422


def test_localize_wrong_content_type():
    """Текстовый файл → 415 Unsupported Media Type."""
    with patch("services.api.routes.localize.localize") as mock_loc:
        resp = client.post(
            "/api/v1/localize",
            files={"image": ("test.txt", b"hello world", "text/plain")},
            data={"top_n": 5},
        )
        assert resp.status_code == 415


def test_localize_index_not_ready():
    """Если индекс не загружен → 503."""
    with patch(
        "services.api.routes.localize.localize",
        side_effect=FileNotFoundError("Index not found"),
    ):
        resp = client.post(
            "/api/v1/localize",
            files={"image": ("photo.jpg", b"\xff\xd8\xff" + b"\x00" * 100, "image/jpeg")},
            data={"top_n": 5},
        )
        assert resp.status_code == 503


def test_admin_index_stats():
    """Stats endpoint возвращает структуру."""
    with (
        patch("services.api.routes.admin.FaissStore.load") as mock_faiss,
        patch("services.api.routes.admin.Vocabulary.load") as mock_vocab,
        patch("services.api.routes.admin.PatchRepo") as mock_repo,
        patch("services.api.routes.admin.SyncSessionLocal"),
    ):
        mock_faiss.return_value.stats.return_value = {
            "ntotal": 50000, "dim": 1024, "n_lists": 256,
            "n_probe": 32, "is_trained": True,
        }
        mock_vocab.return_value.vocab_size = 1024
        mock_repo.return_value.count_patches.return_value = 50000

        resp = client.get("/api/v1/admin/index/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "faiss_ntotal" in data
        assert "vocab_size" in data
