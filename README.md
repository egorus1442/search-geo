# GeoVision

Сервис геолокализации: определяет координаты места съёмки по фотографии с дрона/зонда, сопоставляя её с эталонной базой спутниковых снимков Sentinel-2.

**Вход:** фото с дрона (JPEG/PNG/TIFF)
**Выход:** top-N кандидатов координат (lat/lon) с оценкой уверенности

📖 Подробная инструкция по запуску и использованию — [`docs/SETUP.md`](docs/SETUP.md)
📄 Спецификация API — [`api.json`](api.json) (OpenAPI 3.0)

## Как это работает

Классический CV-пайплайн: SIFT-дескрипторы → Bag of Visual Words → приближённый поиск FAISS → геометрическая проверка RANSAC.

```mermaid
flowchart LR
    subgraph offline["Offline: построение базы"]
        S2[Sentinel-2 тайлы] --> Cut[Нарезка на патчи 256×256]
        Cut --> Feat1[SIFT дескрипторы]
        Feat1 --> Vocab[BoVW словарь]
        Vocab --> Idx[FAISS IVF индекс]
    end

    subgraph online["Online: геолокализация"]
        Photo[Фото с дрона] --> Feat2[SIFT дескрипторы]
        Feat2 --> BoVW2[BoVW гистограмма]
        BoVW2 --> Search[FAISS поиск: top-100]
        Search --> Match[BFMatcher + RANSAC]
        Match --> Result[Top-N координат]
    end

    Idx -.используется.-> Search
    Match --> DB[(PostgreSQL/PostGIS)]
    DB --> Result
```

## Архитектура сервиса

```mermaid
flowchart TB
    Client[Клиент] -->|REST| API[FastAPI]

    API --> DB[(PostgreSQL + PostGIS<br/>тайлы, патчи, задачи)]
    API --> Redis[(Redis<br/>очередь Celery)]
    API --> FS[FAISS индекс + BoVW словарь<br/>файлы на диске]

    Redis --> Worker[Celery Worker]
    Worker --> CDSE[Copernicus CDSE<br/>источник снимков]
    Worker --> Minio[(MinIO<br/>хранилище патчей/снимков)]
    Worker --> DB
    Worker --> FS

    API --> Minio
```

**Основные компоненты:**

| Компонент | Роль |
|---|---|
| `services/api` | FastAPI: приём фото, отдача результатов, админ-эндпоинты |
| `services/ingestor` | Скачивание снимков Sentinel-2 (CDSE), нарезка на патчи, загрузка в MinIO |
| `services/features` | Извлечение SIFT-дескрипторов, обучение BoVW словаря |
| `services/index` | FAISS-индекс (векторный поиск) + метаданные патчей в PostgreSQL |
| `services/matching` | Онлайн-пайплайн локализации: поиск кандидатов + RANSAC-верификация |
| `workers` | Celery-задачи: ingestion и построение индекса (долгие операции) |

## Технологии

FastAPI · OpenCV (SIFT) · FAISS · PostgreSQL/PostGIS · Celery + Redis · MinIO · Docker Compose

## Быстрый старт

```bash
cp .env.example .env        # заполнить CDSE credentials
docker-compose up -d
docker-compose exec api alembic upgrade head
```

Далее — ingestion снимков, построение индекса и запросы геолокализации: см. [`docs/SETUP.md`](docs/SETUP.md).
