# GeoVision

Микросервис геолокализации аэрофотоснимков по спутниковой базе.

**Вход:** фото с дрона/зонда (JPEG/PNG)  
**Выход:** top-N кандидатов с координатами (lat/lon) и оценкой уверенности

## Алгоритм

```
[Эталонная база Sentinel-2]        [Query фото]
   └── SIFT дескрипторы               └── SIFT дескрипторы
   └── BoVW гистограммы               └── BoVW гистограмма
   └── FAISS IVF индекс         ──►   └── FAISS поиск → top-100
                                      └── BFMatcher + RANSAC → top-N
                                      └── Координаты из PostgreSQL
```

## Быстрый старт

### 1. Настройка

```bash
cp .env.example .env
# Отредактировать .env: CDSE_CLIENT_ID, CDSE_CLIENT_SECRET
# Получить credentials: https://dataspace.copernicus.eu/
```

### 2. Запуск сервисов

```bash
docker-compose up -d
```

### 3. Применение миграций

```bash
docker-compose exec api alembic upgrade head
```

### 4. Ingestion (скачать спутниковые снимки)

Через API:
```bash
curl -X POST http://localhost:8000/api/v1/admin/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [35.0, 50.0, 40.0, 55.0],
    "date_from": "2023-06-01",
    "date_to": "2024-09-30",
    "cloud_cover_max": 20
  }'
```

Или через CLI (синхронно, для отладки):
```bash
docker-compose exec api python scripts/ingest_region.py \
  --bbox 35.0 50.0 40.0 55.0 \
  --date-from 2023-06-01 \
  --date-to 2024-09-30
```

### 5. Построение индекса

```bash
# Шаг 1: обучить BoVW словарь
curl -X POST http://localhost:8000/api/v1/admin/index/vocabulary

# Шаг 2: построить FAISS индекс
curl -X POST http://localhost:8000/api/v1/admin/index/build

# Или через CLI:
docker-compose exec api python scripts/build_vocab.py
```

### 6. Геолокализация

```bash
curl -X POST http://localhost:8000/api/v1/localize \
  -F "image=@/path/to/drone_photo.jpg" \
  -F "top_n=10"
```

Или через Python скрипт:
```bash
python scripts/test_localize.py --image drone_photo.jpg
```

## API

| Endpoint | Метод | Описание |
|---|---|---|
| `/api/v1/localize` | POST | Геолокализация фото |
| `/api/v1/admin/ingest` | POST | Запустить скачивание снимков |
| `/api/v1/admin/ingest/{id}` | GET | Статус задачи |
| `/api/v1/admin/index/vocabulary` | POST | Обучить BoVW словарь |
| `/api/v1/admin/index/build` | POST | Построить FAISS индекс |
| `/api/v1/admin/index/reload` | POST | Перезагрузить индекс |
| `/api/v1/admin/index/stats` | GET | Статистика индекса |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## Пример ответа `/api/v1/localize`

```json
{
  "task_id": "uuid",
  "status": "completed",
  "processing_time_ms": 12400,
  "candidates": [
    {
      "rank": 1,
      "patch_id": 42,
      "center_lat": 51.234,
      "center_lon": 36.789,
      "bbox": [36.77, 51.22, 36.80, 51.25],
      "inlier_count": 38,
      "confidence": 0.76,
      "thumbnail_url": "http://minio:9000/geovision/patches/..."
    }
  ]
}
```

## Архитектура

```
geovision/
├── config/              # Pydantic Settings + logging
├── services/
│   ├── db/             # SQLAlchemy модели + сессии
│   ├── ingestor/       # CDSE клиент, tile cutter, MinIO
│   ├── features/       # SIFT, BoVW Vocabulary
│   ├── index/          # FAISS store, PostgreSQL metadata repo
│   ├── matching/       # RANSAC верификатор, online pipeline
│   └── api/            # FastAPI app + routes
├── workers/            # Celery app + задачи
├── migrations/         # Alembic
└── scripts/            # CLI утилиты
```

## Сервисы (Docker Compose)

| Сервис | Порт | Описание |
|---|---|---|
| `api` | 8000 | REST API (FastAPI) |
| `worker` | — | Celery worker |
| `postgres` | 5432 | PostgreSQL + PostGIS |
| `redis` | 6379 | Celery broker |
| `minio` | 9000/9001 | Объектное хранилище / UI |

## Настройка (ключевые параметры `.env`)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `SIFT_N_FEATURES` | 2000 | Макс. ключевых точек SIFT |
| `VOCAB_SIZE` | 1024 | Размер BoVW словаря |
| `FAISS_N_LISTS` | 256 | Ячейки IVF |
| `FAISS_N_PROBE` | 32 | Проверяемых ячеек при поиске |
| `PATCH_SIZE_PX` | 256 | Размер патча в пикселях |
| `PATCH_OVERLAP_RATIO` | 0.25 | Перекрытие патчей |
| `TOP_N_COARSE` | 100 | Кандидаты после FAISS |
| `TOP_N_RESULT` | 10 | Финальные результаты |
| `LOWE_RATIO` | 0.75 | Lowe ratio test |
| `RANSAC_THRESHOLD` | 5.0 | RANSAC reprojection error (px) |
