# Установка и использование

> Полное практическое руководство: настройка, запуск, ingestion, индексирование, геолокализация.
> Общее описание проекта и архитектуры — в [../README.md](../README.md).

## 1. Настройка

```bash
cp .env.example .env
# Отредактировать .env: CDSE_USERNAME, CDSE_PASSWORD (аккаунт dataspace.copernicus.eu)
# Выбрать coarse-метод: COARSE_METHOD=vlad|bovw|dino|dino_vlad (см. README)
```

> Для нейро-методов (`dino`/`dino_vlad`) образ должен быть собран с torch/timm:
> `docker-compose build` (в `Dockerfile` включён `ARG WITH_DINO=1`). Классические
> `vlad`/`bovw` дополнительных зависимостей не требуют.

## 2. Запуск сервисов

```bash
docker-compose up -d
```

## 3. Применение миграций

```bash
docker-compose exec api alembic upgrade head
```

## 4. Ingestion (скачать спутниковые снимки)

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

## 5. Построение индекса

Индекс строится под текущий `COARSE_METHOD` (см. `.env`). При смене метода
пересоберите индекс и перезагрузите.

```bash
# Шаг 1: подготовить/обучить coarse-энкодер (BoVW/VLAD — обучение; dino — no-op)
curl -X POST http://localhost:8000/api/v1/admin/index/vocabulary

# Шаг 2: построить FAISS индекс (для dino/dino_vlad — отдельный GLOBAL_INDEX_PATH)
curl -X POST http://localhost:8000/api/v1/admin/index/build

# Шаг 3: перезагрузить индекс в работающем сервисе
curl -X POST http://localhost:8000/api/v1/admin/index/reload
```

## 6. Геолокализация

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

Полная спецификация: [../api.json](../api.json) (OpenAPI 3.0).

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

## Архитектура каталогов

```
geovision/
├── config/              # Pydantic Settings + logging
├── services/
│   ├── db/             # SQLAlchemy модели + сессии
│   ├── ingestor/       # CDSE клиент, tile cutter, MinIO
│   ├── features/       # SIFT + coarse-энкодеры (BoVW, VLAD, DINOv2, AnyLoc)
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
| `COARSE_METHOD` | vlad | Метод coarse-отбора: `bovw` \| `vlad` \| `dino` \| `dino_vlad` |
| `SIFT_N_FEATURES` | 2000 | Макс. ключевых точек SIFT |
| `VOCAB_SIZE` | 1024 | Размер BoVW словаря |
| `FAISS_N_LISTS` | 256 | Ячейки IVF |
| `FAISS_N_PROBE` | 32 | Проверяемых ячеек при поиске |
| `PATCH_SIZE_PX` | 256 | Размер патча в пикселях |
| `PATCH_OVERLAP_RATIO` | 0.25 | Перекрытие патчей |
| `TOP_N_COARSE` | 100 | Кандидаты после FAISS (для `vlad`/`bovw`) |
| `TOP_N_RESULT` | 10 | Финальные результаты |
| `LOWE_RATIO` | 0.75 | Lowe ratio test |
| `RANSAC_THRESHOLD` | 5.0 | RANSAC reprojection error (px) |

**Нейро-методы (`dino`/`dino_vlad`), ключевые параметры:**

| Переменная | По умолчанию | Описание |
|---|---|---|
| `GLOBAL_MODEL_NAME` | vit_small_patch14_dinov2.lvd142m | Бэкбон DINOv2 (timm) |
| `GLOBAL_IMAGE_SIZE` | 224 | Размер входа сети (кратно 14) |
| `GLOBAL_POOLING` | gem | Пулинг для `dino`: `cls` \| `mean` \| `gem` |
| `GLOBAL_TOP_K` | 300 | Кандидаты после FAISS (для `dino`/`dino_vlad`) |
| `GLOBAL_INDEX_PATH` | /data/index/global_index.faiss | Отдельный FAISS-индекс нейро-методов |
| `TORCH_NUM_THREADS` | 0 | Потоки torch на CPU (0 = авто) |
