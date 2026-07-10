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

    # ── Coarse retrieval (какой глобальный дескриптор кормит FAISS) ─────────────
    # bovw — старый tf-idf Bag of Visual Words (грубый, режет верный патч на
    #        больших площадях, см. историю и честный тест).
    # vlad — VLAD поверх RootSIFT + PCA-whitening: агрегирует остатки к
    #        центроидам, гораздо различимее BoVW, ложится в FAISS так же.
    # dino — DINOv2 (ViT-S/14) глобальный эмбеддинг патча (cls|mean|gem pooling).
    #        Кодирует «место» из САМОЙ картинки, а не из SIFT — устойчивее к
    #        domain gap UAV↔Sentinel и к низкотекстурному контенту (лес/поля).
    # dino_vlad — AnyLoc: VLAD-агрегация DINOv2 patch-токенов. Обычно различимее
    #        плоского pooling'а на больших площадях, ценой размерности вектора.
    # Verifier (SIFT+RANSAC) не зависит от этого выбора — coarse только
    # выбирает кандидатов на дорогую геометрическую проверку.
    # dino/dino_vlad кодируют из КАРТИНКИ (input_kind="image"), vlad/bovw — из
    # SIFT-дескрипторов; index_task/localize диспетчеризуют по input_kind.
    coarse_method: str = "vlad"  # bovw | vlad | dino | dino_vlad

    # ── BoVW ──────────────────────────────────────────────────────────────────
    vocab_size: int = 1024
    vocab_sample_per_patch: int = 50
    vocab_kmeans_n_init: int = 10

    # ── VLAD ──────────────────────────────────────────────────────────────────
    # RootSIFT (Hellinger-норм: L1 → sqrt) применяется внутри энкодера к
    # дескрипторам ДО агрегации — дешёвый и стабильный прирост recall, не
    # затрагивает raw-SIFT дескрипторы, которые уходят в verifier.
    vlad_n_centroids: int = 64          # размер VLAD = n_centroids * 128 до PCA
    vlad_sample_per_patch: int = 100    # сколько дескрипторов брать на патч для k-means
    vlad_kmeans_n_init: int = 3
    vlad_use_pca: bool = True           # PCA-whitening финального VLAD-вектора
    vlad_pca_dim: int = 256             # итоговая размерность после PCA-whitening
    vlad_path: Path = Path("/data/index/vlad.pkl")

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_n_lists: int = 256
    faiss_n_probe: int = 32
    faiss_index_path: Path = Path("/data/index/patch_index.faiss")
    vocab_path: Path = Path("/data/index/vocabulary.pkl")

    # ── Global descriptors (DINOv2 / AnyLoc) ──────────────────────────────────
    # Нейросетевой coarse-дескриптор (COARSE_METHOD=dino|dino_vlad). torch/timm —
    # опциональные зависимости, импортируются лениво только под эти методы (см.
    # services/features/dino.py). Веса timm скачиваются в кэш при первом запуске.
    global_model_name: str = "vit_small_patch14_dinov2.lvd142m"
    global_descriptor_dim: int = 384          # embed_dim бэкбона (dim для pooling-голов)
    global_image_size: int = 224              # кратно patch=14; больше → точнее, но дороже CPU
    global_pooling: str = "gem"               # cls | mean | gem (для method=dino)
    global_gem_p: float = 3.0                 # степень GeM-pooling
    global_index_path: Path = Path("/data/index/global_index.faiss")  # отдельный FAISS
    global_model_path: Path = Path("/data/index/global_encoder.pkl")  # конфиг/словарь энкодера
    global_top_k: int = 300                   # k FAISS-кандидатов на verifier (image-методы)
    # Ниже этого числа патчей глобальный индекс строится как Flat (точный),
    # а не IVF — на малой базе IVF режет верный ответ (см. историю coarse-провала).
    global_use_ivf_threshold: int = 50000
    # AnyLoc (dino_vlad): VLAD поверх DINOv2 patch-токенов.
    global_vlad_n_centroids: int = 32         # dim = n_centroids * global_descriptor_dim
    global_vlad_sample_per_patch: int = 100   # токенов на патч для обучения k-means
    global_vlad_kmeans_n_init: int = 3
    # torch: 0 = авто (не трогаем set_num_threads), >0 = зафиксировать число потоков.
    torch_num_threads: int = 0

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

    # ── Photometric check (NCC после варпа) ──────────────────────────────────
    # Доп. независимый фильтр ПОВЕРХ RANSAC: используя affine-матрицу,
    # найденную RANSAC'ом, варпим query в систему координат кандидата и
    # считаем normalized cross-correlation (NCC) на области перекрытия.
    # Идея: inlier_count/inlier_ratio — чисто геометрический сигнал, его можно
    # "обмануть" повторяющейся текстурой (ряды деревьев, поля, регулярная
    # застройка) — геометрия согласуется, а по факту это другой участок.
    # NCC — ортогональный сигнал ("действительно ли пиксели похожи после
    # совмещения"), не связанный с тем, сколько точек согласовал RANSAC.
    photometric_check_enabled: bool = False
    # Если после варпа валидная (не выходящая за пределы query) область
    # кандидата меньше этой доли — NCC не считается (слишком мало данных,
    # неинформативно), кандидат получает нейтральный фактор.
    photometric_min_overlap_ratio: float = 0.3

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
