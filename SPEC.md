# GeoVision — Микросервис геолокализации по аэрофотоснимку

**Версия:** 1.0 MVP  
**Дата:** 2026-06-30  
**Цель:** по фото с дрона/зонда → вернуть top-N кандидатов с координатами

---

## 1. Обзор системы

### 1.1 Задача

Входное фото (с дрона/зонда, смешанная высота 50 м – десятки км) сопоставляется с эталонной базой геопривязанных спутниковых патчей. На выходе — список кандидатов с координатами центра и оценкой уверенности.

**Точность MVP:** 1–10 км (определяется размером патча и качеством совпадения)

### 1.2 Ограничения MVP

- Покрытие: настраиваемый bbox (цель — приграничные регионы России)
- Источник: Sentinel-2 L2A (Copernicus, 10 м/пкс, бесплатно, без ToS-ограничений)
- Алгоритм: классический CV (SIFT + BoVW + FAISS + RANSAC)
- Инфраструктура: CPU-only, Docker Compose

---

## 2. Архитектура

### 2.1 Компоненты

```
┌──────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                      │
│                                                          │
│  [CDSE API] ──► [Ingestor] ──► [Tile Storage]           │
│                                    │                     │
│                          [Feature Extractor]             │
│                          ├── SIFT keypoints              │
│                          └── BoVW encoding               │
│                                    │                     │
│                          [Indexer]                       │
│                          ├── FAISS index build           │
│                          └── PostgreSQL metadata         │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    ONLINE PIPELINE                       │
│                                                          │
│  [Client] ──► POST /api/v1/localize                      │
│                    │                                     │
│              [API Service]                               │
│                    │                                     │
│              [Query Processor]                           │
│              ├── SIFT extraction                         │
│              ├── BoVW encoding                           │
│              └── FAISS search (top-100)                  │
│                    │                                     │
│              [Verifier]                                  │
│              ├── BFMatcher per candidate                 │
│              ├── RANSAC homography                       │
│              └── inlier count scoring                    │
│                    │                                     │
│              [Result Builder]                            │
│              └── top-N + координаты из PostgreSQL        │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Сервисы (Docker Compose)

| Сервис | Образ | Роль |
|---|---|---|
| `api` | Python 3.11 / FastAPI | REST endpoint, оркестрация |
| `worker` | Python 3.11 / Celery | Async задачи: ingestion, indexing |
| `postgres` | postgres:16-alpine + PostGIS | Метаданные патчей (id, bbox, координаты) |
| `redis` | redis:7-alpine | Celery broker + кэш результатов |
| `minio` | minio/minio | S3-совместимое хранилище тайлов |

### 2.3 Структура проекта

```
geovision/
├── docker-compose.yml
├── .env.example
├── config/
│   └── settings.py          # Pydantic Settings
├── services/
│   ├── ingestor/            # Скачивание Sentinel-2
│   │   ├── cdse_client.py   # CDSE OData API
│   │   └── tile_cutter.py   # rasterio: нарезка патчей
│   ├── features/            # Извлечение признаков
│   │   ├── sift.py          # SIFT keypoints + descriptors
│   │   ├── vocabulary.py    # K-Means BoVW словарь
│   │   └── encoder.py       # Патч → BoVW гистограмма (tf-idf)
│   ├── index/               # Индексирование
│   │   ├── faiss_store.py   # FAISS IVF-Flat / IVF-PQ
│   │   └── metadata_store.py # PostgreSQL: patch_id → координаты
│   ├── matching/            # Верификация кандидатов
│   │   ├── retrieval.py     # FAISS поиск top-100
│   │   └── verifier.py      # BFMatcher + RANSAC
│   └── api/
│       ├── main.py          # FastAPI приложение
│       ├── routes/
│       │   ├── localize.py  # POST /api/v1/localize
│       │   ├── admin.py     # POST /api/v1/admin/ingest
│       │   └── health.py    # GET /health
│       └── schemas.py       # Pydantic модели запрос/ответ
├── workers/
│   ├── celery_app.py
│   └── tasks/
│       ├── ingest_task.py   # Задача: скачать + нарезать регион
│       └── index_task.py    # Задача: построить/обновить индекс
├── migrations/              # Alembic
├── tests/
│   ├── unit/
│   └── integration/
└── scripts/
    ├── build_vocab.py       # Оффлайн: обучить BoVW словарь
    └── ingest_region.py     # CLI: запустить ingestion по bbox
```

---

## 3. Алгоритм (пошагово)

### 3.1 Offline: формирование эталонной базы

#### Шаг 1 — Ingestion (скачивание спутниковых снимков)

- API: Copernicus Data Space Ecosystem (CDSE) OData
- Запрос: по bbox + дата + cloud_cover < 20%
- Продукт: Sentinel-2 L2A (каналы B04/B03/B02 → RGB)
- Токен: OAuth2 client credentials (CDSE dashboard)
- Библиотека: `sentinelhub` или прямые HTTP-запросы

**Параметры:**
```yaml
region_bbox: [lon_min, lat_min, lon_max, lat_max]
cloud_cover_max: 20        # %
date_from: "2023-06-01"
date_to: "2024-09-30"      # летний период для чистых снимков
```

#### Шаг 2 — Tile Cutting (нарезка патчей)

Скачанные .SAFE → rasterio → патчи PNG

**Параметры:**
```yaml
patch_size_px: 256          # пикселей
gsd_target_m: 10            # м/пкс (нативный Sentinel-2)
overlap_ratio: 0.25         # 25% перекрытие
min_coverage: 0.8           # патч отбрасывается если > 20% nodata
```

**Один патч покрывает:** 256 × 10 м = **2.56 км × 2.56 км**  
**Оценка объёма для 500 000 км²:** ~100 000–150 000 патчей

Каждый патч сохраняется в MinIO. В PostgreSQL пишется запись:
```
patch_id | source_tile_id | center_lat | center_lon | bbox_wkt | s3_path | created_at
```

#### Шаг 3 — Feature Extraction (SIFT)

Для каждого патча:
- `cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)`
- Извлечение `keypoints` + `descriptors` (128-dim float32)
- Дескрипторы сохраняются как numpy `.npy` в MinIO (или встраиваются в индекс)

#### Шаг 4 — BoVW Vocabulary Training (K-Means)

Выполняется один раз при первичной сборке:

```python
# Сэмплируем N=500_000 дескрипторов из всей базы
vocab = KMeans(n_clusters=1024, n_init=10)
vocab.fit(sampled_descriptors)
# Сохраняем центроиды → vocabulary.pkl
```

**Параметры:**
```yaml
vocab_size: 1024            # число визуальных слов
kmeans_n_init: 10
```

#### Шаг 5 — BoVW Encoding

Каждый патч → гистограмма 1024-dim с tf-idf взвешиванием:

```python
def encode(descriptors, vocab, idf_weights):
    word_ids = vocab.predict(descriptors)       # ближайшее слово
    hist = np.bincount(word_ids, minlength=1024).astype(float)
    hist *= idf_weights                         # tf-idf
    hist /= np.linalg.norm(hist) + 1e-8        # L2-нормировка
    return hist.astype(np.float32)
```

#### Шаг 6 — FAISS Indexing

```python
import faiss
# MVP: IVF-Flat (точный, умеренная скорость)
quantizer = faiss.IndexFlatL2(1024)
index = faiss.IndexIVFFlat(quantizer, 1024, n_lists=256)
index.train(all_histograms)
index.add_with_ids(all_histograms, patch_ids)
faiss.write_index(index, "patch_index.faiss")
```

**Параметры:**
```yaml
faiss_n_lists: 256          # число ячеек IVF
faiss_n_probe: 32           # число проверяемых ячеек при поиске
```

**Размер индекса:** 150 000 патчей × 1024 × 4 байт ≈ **600 МБ** (в RAM)

---

### 3.2 Online: запрос геолокализации

#### Шаг 1 — Preprocessing

```
POST /api/v1/localize
Content-Type: multipart/form-data
Body: image=<файл>, top_n=10
```

- Принять изображение (JPEG/PNG, до 50 МБ)
- Ресайз до 1024px по длинной стороне (опционально)
- Конвертация в grayscale для SIFT

#### Шаг 2 — Query Feature Extraction

- SIFT с теми же параметрами, что и для базы
- Encode в BoVW гистограмму

#### Шаг 3 — Coarse Retrieval (FAISS)

```python
D, I = index.search(query_hist.reshape(1, -1), k=100)
# D: L2 distances, I: patch_ids
candidate_ids = I[0]  # top-100
```

#### Шаг 4 — Fine Verification (RANSAC)

Для каждого из top-100 кандидатов:

```python
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(q_desc, cand_desc, k=2)
# Ratio test (Lowe)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
# RANSAC homography
if len(good) >= 4:
    H, mask = cv2.findHomography(q_pts, c_pts, cv2.RANSAC, 5.0)
    inliers = mask.sum()
```

Сортировка по `inliers` DESC → top-N результатов

#### Шаг 5 — Result Building

```python
# Из PostgreSQL получаем координаты для top-N patch_id
result = [
    {
        "rank": i+1,
        "patch_id": pid,
        "center_lat": lat,
        "center_lon": lon,
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "inlier_count": n_inliers,
        "confidence": min(n_inliers / 50.0, 1.0)
    }
    for i, (pid, n_inliers) in enumerate(top_n)
]
```

---

## 4. API

### 4.1 Endpoints

| Method | Path | Описание |
|---|---|---|
| `POST` | `/api/v1/localize` | Геолокализация по фото |
| `GET` | `/api/v1/localize/{task_id}` | Статус/результат задачи |
| `POST` | `/api/v1/admin/ingest` | Запустить ingestion по bbox |
| `GET` | `/api/v1/admin/ingest/{task_id}` | Статус ingestion задачи |
| `GET` | `/api/v1/admin/index/stats` | Статистика индекса |
| `GET` | `/health` | Health check |

### 4.2 Схемы

**Request: POST /api/v1/localize**
```json
{
  "image": "<multipart file>",
  "top_n": 10
}
```

**Response: GET /api/v1/localize/{task_id}**
```json
{
  "task_id": "uuid",
  "status": "completed",
  "processing_time_ms": 4200,
  "candidates": [
    {
      "rank": 1,
      "patch_id": "abc123",
      "center_lat": 51.234,
      "center_lon": 36.789,
      "bbox": [36.77, 51.22, 36.80, 51.25],
      "inlier_count": 42,
      "confidence": 0.84,
      "thumbnail_url": "http://minio/patches/abc123.png"
    }
  ]
}
```

**Request: POST /api/v1/admin/ingest**
```json
{
  "bbox": [35.0, 50.0, 40.0, 55.0],
  "date_from": "2023-06-01",
  "date_to": "2024-09-30",
  "cloud_cover_max": 20
}
```

---

## 5. Стек технологий

| Категория | Технология | Версия |
|---|---|---|
| API Framework | FastAPI | ≥ 0.111 |
| Async Tasks | Celery | ≥ 5.4 |
| Message Broker | Redis | 7.x |
| Feature Extraction | OpenCV-contrib | ≥ 4.10 |
| Vector Index | FAISS-cpu | ≥ 1.8 |
| BoVW Clustering | scikit-learn | ≥ 1.5 |
| Satellite Imagery | sentinelhub | ≥ 3.10 |
| Raster Processing | rasterio + GDAL | ≥ 1.3 |
| Geo DB | PostgreSQL 16 + PostGIS 3 | — |
| ORM | SQLAlchemy 2.x + asyncpg | — |
| Migrations | Alembic | — |
| Object Storage | MinIO (S3-совместимый) | — |
| Config | Pydantic Settings v2 | — |
| Validation | Pydantic v2 | — |
| Logging | structlog | — |
| Metrics | prometheus-client | — |
| Containerization | Docker + Docker Compose | — |
| Testing | pytest + pytest-asyncio | — |

---

## 6. База данных

### 6.1 PostgreSQL схема

```sql
-- Источниковые тайлы Sentinel-2
CREATE TABLE source_tiles (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id  TEXT NOT NULL,           -- CDSE product ID
    bbox        GEOMETRY(POLYGON, 4326),
    date_acq    DATE NOT NULL,
    cloud_cover FLOAT,
    s3_path     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Патчи (эталонная база)
CREATE TABLE patches (
    id          BIGSERIAL PRIMARY KEY,
    source_tile UUID REFERENCES source_tiles(id),
    center      GEOMETRY(POINT, 4326) NOT NULL,
    bbox        GEOMETRY(POLYGON, 4326) NOT NULL,
    s3_path     TEXT NOT NULL,
    patch_size  INT DEFAULT 256,
    gsd_m       FLOAT DEFAULT 10.0,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON patches USING GIST (center);
CREATE INDEX ON patches USING GIST (bbox);

-- Задачи (ingestion, indexing)
CREATE TABLE tasks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type        TEXT NOT NULL,           -- 'ingest' | 'index'
    status      TEXT NOT NULL,           -- 'pending'|'running'|'done'|'failed'
    params      JSONB,
    result      JSONB,
    error       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
```

---

## 7. Конфигурация

```env
# .env
DATABASE_URL=postgresql+asyncpg://geo:geo@postgres:5432/geovision
REDIS_URL=redis://redis:6379/0
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=geovision

# CDSE Credentials
CDSE_CLIENT_ID=...
CDSE_CLIENT_SECRET=...

# Algorithm params
SIFT_N_FEATURES=2000
SIFT_CONTRAST_THRESHOLD=0.04
VOCAB_SIZE=1024
FAISS_N_LISTS=256
FAISS_N_PROBE=32
PATCH_SIZE_PX=256
PATCH_OVERLAP_RATIO=0.25
RANSAC_THRESHOLD=5.0
LOWE_RATIO=0.75
TOP_N_COARSE=100
TOP_N_RESULT=10
```

---

## 8. Производительность (оценка для MVP)

| Метрика | Значение |
|---|---|
| Покрытие | ~500 000 км² (настраивается) |
| Кол-во патчей | ~100 000–150 000 |
| Размер FAISS индекса | ~600 МБ RAM |
| FAISS поиск top-100 | < 50 мс |
| RANSAC × 100 кандидатов | 5–30 сек (CPU) |
| Общее время запроса | ~10–60 сек (batch — приемлемо) |
| Время полного ingestion | 2–8 часов (зависит от региона) |

---

## 9. Этапы реализации (фазы)

### Фаза 1 — Core Pipeline (Недели 1–2)
- [ ] Структура проекта + Docker Compose
- [ ] CDSE клиент: поиск и скачивание Sentinel-2 по bbox
- [ ] Tile Cutter: rasterio → патчи PNG в MinIO
- [ ] PostgreSQL схема + Alembic миграции
- [ ] SIFT extraction + BoVW vocabulary training (скрипт)
- [ ] FAISS index builder

### Фаза 2 — API (Неделя 3)
- [ ] FastAPI приложение: `/localize`, `/health`
- [ ] Celery задачи: ingest_task, index_task
- [ ] Полный online pipeline (SIFT → BoVW → FAISS → RANSAC)
- [ ] Admin endpoints: `/admin/ingest`, `/admin/index/stats`

### Фаза 3 — Качество и наблюдаемость (Неделя 4)
- [ ] Структурированное логирование (structlog)
- [ ] Метрики (Prometheus)
- [ ] Unit + integration тесты
- [ ] Документация API (автоматически через FastAPI)
- [ ] Скрипт первичного запуска (`scripts/ingest_region.py`)

---

## 10. Ограничения и риски MVP

| Риск | Описание | Митигация |
|---|---|---|
| Точность BoVW | SIFT + BoVW хуже DL при сильном изменении ракурса/масштаба | Увеличить vocab_size, настроить nfeatures |
| Размер базы | >500К патчей → FAISS не помещается в RAM | IVF-PQ (квантизация), или Qdrant в Фазе 2 |
| CDSE ToS | Лимиты на скачивание | Кэшировать тайлы, incremental updates |
| Масштаб России | Страна огромная — MVP на фиксированный bbox | Config-driven, iterative expansion |
| Нет облаков → нет снимков | Регионы с постоянной облачностью | Расширить date range, temporal composite |

---

## 11. Будущие улучшения (после MVP)

1. **DL-гибрид**: заменить BoVW на AnyLoc-VLAD-DINOv2 → +30-50% точности
2. **SuperPoint + LightGlue**: заменить SIFT+RANSAC для верификации
3. **Qdrant**: заменить FAISS для управляемого хранения + geo-фильтрации
4. **Temporal fusion**: несколько снимков одного района → более чистая база
5. **Confidence calibration**: исторические данные для калибровки confidence score
