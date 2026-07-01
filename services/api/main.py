from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import configure_logging, get_settings
from services.api.routes.admin import router as admin_router
from services.api.routes.health import router as health_router
from services.api.routes.localize import router as localize_router
from services.ingestor.storage import ensure_bucket

_s = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(_s.log_level)
    try:
        ensure_bucket()
    except Exception:
        pass  # Не падать при старте если MinIO ещё не готов
    yield


app = FastAPI(
    title="GeoVision",
    description="Геолокализация аэрофотоснимков по базе спутниковых тайлов",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(localize_router)
app.include_router(admin_router)
