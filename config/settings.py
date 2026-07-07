from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://geo:geo@localhost:5432/geovision"
    database_url_sync: str = "postgresql+psycopg2://geo:geo@localhost:5432/geovision"

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── MinIO ─────────────────────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "geovision"
    minio_secure: bool = False

    # ── CDSE ──────────────────────────────────────────────────────────────────
    # Логин и пароль от аккаунта dataspace.copernicus.eu
    # Создавать отдельный OAuth client не нужно
    cdse_username: str = ""
    cdse_password: str = ""

    # ── SIFT ──────────────────────────────────────────────────────────────────
    sift_n_features: int = 2000
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: int = 10
    sift_sigma: float = 1.6

    # ── Preprocessing (эксперимент: сокращение domain gap UAV↔Sentinel-2) ───────
    # По умолчанию всё выключено (=текущее поведение). Включать через .env для
    # экспериментов — ВАЖНО: при изменении нужно пересобрать словарь и индекс,
    # т.к. предобработка одинаково влияет и на патчи базы, и на query.
    preprocess_resize_scale: float = 1.0       # <1.0 — понизить разрешение перед SIFT
    preprocess_query_resize_scale: float | None = None  # если задано, scale только для query/UAV
    preprocess_patch_resize_scale: float | None = None  # если задано, scale только для Sentinel-патчей
    preprocess_normalize_channels: bool = False  # percentile-стретч по каналам
    preprocess_use_clahe: bool = False           # адаптивное выравнивание контраста
    preprocess_use_lcn: bool = False             # локальная нормализация контраста ("норм. карта")

    # ── BoVW ──────────────────────────────────────────────────────────────────
    vocab_size: int = 1024
    vocab_sample_per_patch: int = 50
    vocab_kmeans_n_init: int = 10

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_n_lists: int = 256
    faiss_n_probe: int = 32
    faiss_index_path: Path = Path("/data/index/patch_index.faiss")
    vocab_path: Path = Path("/data/index/vocabulary.pkl")

    # ── Tiling ────────────────────────────────────────────────────────────────
    patch_size_px: int = 256
    patch_overlap_ratio: float = 0.25
    patch_min_coverage: float = 0.8
    patch_gsd_m: float = 10.0

    # ── Matching ──────────────────────────────────────────────────────────────
    ransac_threshold: float = 5.0
    lowe_ratio: float = 0.75
    top_n_coarse: int = 100
    top_n_result: int = 10
    min_good_matches: int = 4

    # Временный флаг: пока база маленькая (тысячи патчей), BoVW/FAISS coarse-этап
    # режет верный ответ до RANSAC-верификации (см. честный тест). При малом N
    # прямой перебор (SIFT+RANSAC по ВСЕМ патчам) быстрее и точнее, чем плохо
    # откалиброванный BoVW-фильтр. Вернуть False, когда база вырастет настолько,
    # что exhaustive-перебор станет медленным (десятки-сотни тысяч патчей+).
    exhaustive_search: bool = False

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
    max_image_size_mb: int = 50
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
